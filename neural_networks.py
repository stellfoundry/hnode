import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """Dynamic Tanh (DyT) - A learnable tanh-based replacement for LayerNorm

    Recent work (March 2025) shows this can match or exceed LayerNorm performance
    while being computationally simpler.
    """

    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias


class SelfAttnEncoderLayer_v2(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout_rate,
        activation,
        use_norms=True,
        use_causal_mask=True,
        ffn_expansion=4,  # Increased from 2
        use_pre_norm=True,  # Pre-norm is more stable
        norm_type="dyntanh",  # "layernorm" or "dyntanh" (dynamic tanh)
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_norms = use_norms
        self.use_causal_mask = use_causal_mask
        self.use_pre_norm = use_pre_norm

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            bias=False,  # Can help optimization
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * ffn_expansion, d_model),
        )

        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

        if use_norms:
            # Option 1: Traditional LayerNorm
            if norm_type == "layernorm":
                self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
                self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
            # Option 2: Dynamic Tanh (DyT) - recent breakthrough replacement
            elif norm_type == "dyntanh":
                self.norm1 = DynamicTanh(d_model, alpha_init=0.5)
                self.norm2 = DynamicTanh(d_model, alpha_init=0.5)

        # Optional: learnable residual scaling for very deep networks
        # self.attn_scale = nn.Parameter(torch.ones(1) * 0.5)
        # self.ffn_scale = nn.Parameter(torch.ones(1) * 0.5)

    def update_dropout_rate(self, new_rate):
        self.dropout_rate = new_rate
        self.multihead_attn.dropout = new_rate
        self.dropout_attn.p = new_rate
        self.dropout_ffn.p = new_rate
        for module in self.ffn:
            if isinstance(module, nn.Dropout):
                module.p = new_rate

    def _get_causal_mask(self, seq_len, device, dtype):
        """Generate causal mask with correct dtype and device"""
        return torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device, dtype=dtype
        )

    def forward(self, x, z=None):
        # input shapes:
        # x: [batch_size, seq_len, d_model] or [seq_len, d_model]

        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]

        # Self-attention with pre/post norm
        residual = x

        if self.use_norms:
            if self.use_pre_norm:
                x_norm = self.norm1(x)
            else:
                x_norm = x
        else:
            x_norm = x

        # Generate causal mask if needed
        attn_mask = None
        if self.use_causal_mask:
            attn_mask = self._get_causal_mask(seq_len, x.device, x.dtype)

        # Attention
        attn_output, _ = self.multihead_attn(
            query=x_norm,
            key=z if z is not None else x_norm,
            value=z if z is not None else x_norm,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=self.use_causal_mask and z is None,  # Only for self-attention
        )

        # First residual connection
        x = residual + self.dropout_attn(attn_output)

        # Post-norm for first sublayer (if using post-norm)
        if self.use_norms and not self.use_pre_norm:
            x = self.norm1(x)

        # FFN with pre/post norm
        residual = x

        if self.use_norms:
            if self.use_pre_norm:
                x_norm = self.norm2(x)
            else:
                x_norm = x
        else:
            x_norm = x

        ffn_output = self.ffn(x_norm)

        # Second residual connection
        x = residual + self.dropout_ffn(ffn_output)

        # Post-norm for second sublayer (if using post-norm)
        if self.use_norms and not self.use_pre_norm:
            x = self.norm2(x)

        return x


class CrossAttnTransformer(nn.Module):
    def __init__(
        self,
        problem_dim=1,
        num_blocks=3,
        latent_dim=32 * 4,
        num_heads=4,
        dropout_rate=0,
        activation=nn.Tanh(),
        use_norms=True,
        z_dim=None,
        use_u_init_embedding=False,
        norm_type="dyntanh",
        use_self_attention_z_block=False,
    ):
        super().__init__()

        self.problem_dim = problem_dim
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_norms = use_norms
        self.output_dim = problem_dim
        self.z_dim = z_dim
        self.use_u_init_embedding = use_u_init_embedding

        self.J_embedding = nn.Linear(self.problem_dim, self.latent_dim)
        self.u_embedding = nn.Linear(self.problem_dim, self.latent_dim)
        self.register_buffer("left", torch.tensor([-1.0]))
        self.register_buffer("center", torch.tensor([0.0]))
        self.register_buffer("right", torch.tensor([1.0]))
        self.time_embedding = nn.Linear(1, self.latent_dim)
        if self.z_dim is not None:
            self.z_embedding = nn.Linear(self.z_dim, self.z_dim * self.latent_dim)
        self.self_attention_z_block = SelfAttnEncoderLayer_v2(
            d_model=self.latent_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            activation=activation,
            use_norms=use_norms,
            use_causal_mask=False,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                SelfAttnEncoderLayer_v2(
                    d_model=self.latent_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    use_norms=use_norms,
                    use_causal_mask=False,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_projection = nn.Linear(self.latent_dim, self.output_dim)

    def update_dropout_rate(self, new_rate):
        self.dropout_rate = new_rate
        for block in self.transformer_blocks:
            block.update_dropout_rate(new_rate)

    def forward_original(self, J_hat, u_hat):
        # J_hat: [batch_size (optional), N_t, problem_dim]
        # u_hat: [batch_size (optional), N_t - 1, problem_dim]
        # embed J_hat and u_hat
        J_emb = self.J_embedding(J_hat)
        u_emb = self.u_embedding(u_hat)

        left_emb = self.time_embedding(self.left).unsqueeze(0)
        center_emb = self.time_embedding(self.center).unsqueeze(0)
        right_emb = self.time_embedding(self.right).unsqueeze(0)

        z_tokens = torch.stack(
            [J_emb[:-1] + left_emb, u_emb + center_emb, J_emb[1:] + right_emb]
        ).transpose(0, 1)

        query_tokens = u_emb.unsqueeze(1)
        for block in self.transformer_blocks:
            query_tokens = block(query_tokens, z=z_tokens)

        return self.final_projection(query_tokens.squeeze(1))

    def forward(self, J_hat, u_hat, z=None):
        # J_hat: [batch_size (optional), N_t, problem_dim]
        # u_hat: [batch_size (optional), N_t - 1, problem_dim]
        # embed J_hat and u_hat
        batch_size = J_hat.shape[0]
        u_hat_size = u_hat.shape[1]

        z_tokens = (
            torch.stack(
                [
                    J_hat[:, :-1],
                    J_hat[:, 1:],
                ]
            )
            .permute(1, 2, 0, 3)
            .reshape(batch_size * u_hat_size, 2, self.problem_dim)
        )
        u_hat = u_hat.reshape(batch_size * u_hat_size, self.problem_dim)

        if z is not None:
            if self.use_u_init_embedding:
                expand_size = self.z_dim + self.problem_dim
            else:
                expand_size = self.z_dim
            z = (z.unsqueeze(1).expand(batch_size, u_hat_size, expand_size)).reshape(
                -1, expand_size
            )

        left_emb = self.time_embedding(self.left.clone())
        center_emb = self.time_embedding(self.center.clone())
        right_emb = self.time_embedding(self.right.clone())

        def internal_forward(u_hat, z_tokens, z=None):
            query_tokens = self.u_embedding(u_hat).unsqueeze(0)
            z_tokens = torch.stack(
                [
                    self.u_embedding(u_hat) + center_emb,
                    self.J_embedding(z_tokens[0]) + left_emb,
                    self.J_embedding(z_tokens[1]) + right_emb,
                ]
            )
            if z is not None and not self.use_u_init_embedding:
                z_emb = self.z_embedding(z).reshape(self.z_dim, self.latent_dim)
                z_tokens = torch.cat([z_tokens, z_emb], dim=0)
                if hasattr(self, "self_attention_z_block"):
                    z_tokens = self.self_attention_z_block(z_tokens)
            elif z is not None and self.use_u_init_embedding:
                z_emb = self.z_embedding(z[: self.z_dim]).reshape(
                    self.z_dim, self.latent_dim
                )
                u_init_emb = self.u_init_embedding(z[self.z_dim :]).unsqueeze(0)
                z_tokens = torch.cat([z_tokens, z_emb, u_init_emb], dim=0)
            for block in self.transformer_blocks:
                query_tokens = block(query_tokens, z=z_tokens)
            u_hat_forcing = self.final_projection(query_tokens.squeeze(0))
            return u_hat_forcing, u_hat_forcing

        if z is None:
            (u_hat_jacobian, z_jacobians), u_hat_forcing = torch.vmap(
                torch.func.jacrev(internal_forward, has_aux=True, argnums=(0, 1)),
                randomness="different",
            )(u_hat, z_tokens)
        else:
            (u_hat_jacobian, z_jacobians), u_hat_forcing = torch.vmap(
                torch.func.jacrev(internal_forward, has_aux=True, argnums=(0, 1)),
                randomness="different",
            )(u_hat, z_tokens, z)

        u_hat_jacobian = u_hat_jacobian.reshape(
            batch_size, u_hat_size, *u_hat_jacobian.shape[1:]
        )
        z_jacobians = z_jacobians.reshape(
            batch_size, u_hat_size, *z_jacobians.shape[1:]
        )
        u_hat_forcing = u_hat_forcing.reshape(
            batch_size, u_hat_size, *u_hat_forcing.shape[1:]
        )

        return (
            (u_hat_jacobian, z_jacobians),
            u_hat_forcing,
        )
