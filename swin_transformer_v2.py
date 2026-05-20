from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ------------------------------
# Utility: DropPath (stochastic depth)
# ------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ------------------------------
# 2D Relative Position Bias (per window)
# ------------------------------
class RelativePositionBias2D(nn.Module):
    def __init__(self, window_size: Tuple[int, int], num_heads: int):
        super().__init__()
        Wh, Ww = window_size
        self.window_size = (Wh, Ww)
        self.num_heads = num_heads
        num_rel = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel, num_heads)
        )

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
        coords_flat = coords.flatten(1)  # 2, Wh*Ww
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        rel_coords[:, :, 0] += Wh - 1
        rel_coords[:, :, 1] += Ww - 1
        rel_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = rel_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        Wh, Ww = self.window_size
        N = Wh * Ww
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()  # h, N, N
        return bias


# ------------------------------
# Window ops with safety checks
# ------------------------------
def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """(B, H, W, C) -> (nW*B, Wh*Ww, C)"""
    B, H, W, C = x.shape
    Wh, Ww = window_size
    assert H % Wh == 0 and W % Ww == 0, "H/W must be divisible by window size"
    x = x.view(B, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, Wh * Ww, C)
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: Tuple[int, int], H: int, W: int
) -> torch.Tensor:
    """(nW*B, Wh*Ww, C) -> (B, H, W, C)"""
    Wh, Ww = window_size
    assert H % Wh == 0 and W % Ww == 0, "H/W must be divisible by window size"
    num_w_h, num_w_w = H // Wh, W // Ww
    num_windows = num_w_h * num_w_w
    B = windows.shape[0] // num_windows
    x = windows.view(B, num_w_h, num_w_w, Wh, Ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ------------------------------
# Attention (SDPA + RPB + optional shifted-window mask)
# ------------------------------
class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = attn_dropout
        self.proj_drop = nn.Dropout(proj_dropout)

        self.rpb = RelativePositionBias2D(window_size, num_heads)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, h, N, d)

        q = q.reshape(-1, N, self.head_dim)
        k = k.reshape(-1, N, self.head_dim)
        v = v.reshape(-1, N, self.head_dim)

        bias = self.rpb()  # (h, N, N)
        bias = bias.unsqueeze(0).repeat(B_, 1, 1, 1).reshape(-1, N, N)  # (B_*h, N, N)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            B = B_ // nW
            mask_exp = attn_mask.unsqueeze(0).repeat(B, 1, 1, 1).view(B_, N, N)
            mask_exp = (
                mask_exp.unsqueeze(1).repeat(1, self.num_heads, 1, 1).view(-1, N, N)
            )
            attn_bias = bias + mask_exp
        else:
            attn_bias = bias

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = (
            x.reshape(B_, self.num_heads, N, self.head_dim)
            .transpose(1, 2)
            .reshape(B_, N, C)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------
# Swin Block (attention + MLP with residuals, plus optional shifted-window masking)
# ------------------------------
class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (8, 8),
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, attn_dropout=attn_drop, proj_dropout=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        self._attn_mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    @torch.no_grad()
    def _compute_attn_mask(
        self, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if self.shift_size == (0, 0):
            return None
        key = (H, W)
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]
        Wh, Ww = self.window_size
        sh, sw = self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        h_slices = (slice(0, -Wh), slice(-Wh, -sh), slice(-sh, None))
        w_slices = (slice(0, -Ww), slice(-Ww, -sw), slice(-sw, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, Wh * Ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(
            attn_mask == 0, 0.0
        )
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size != (0, 0):
            x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)
            )

        x_windows = window_partition(x, self.window_size)
        attn_mask = self._compute_attn_mask(H, W, x_windows.device)
        x_windows = self.attn(x_windows, attn_mask)

        x = window_reverse(
            x_windows.view(-1, self.window_size[0] * self.window_size[1], C),
            self.window_size,
            H,
            W,
        )

        if self.shift_size != (0, 0):
            x = torch.roll(
                x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2)
            )

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ------------------------------
# Patch embed / merge / expand (LayerNorms)
# ------------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 512),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        return x


class PatchMerge(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.expand(x)  # (B, L, 2C)
        x = x.view(B, H, W, 2 * C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x


# ------------------------------
# Tiny MLP (for latent head/tail)
# ------------------------------
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Optional[int] = None,
        out_dim: Optional[int] = None,
        p: float = 0.0,
    ):
        super().__init__()
        hidden = hidden or in_dim
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
            nn.Dropout(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------
# Encoder / Decoder (image-only) with latent head/tail MLPs
# ------------------------------
class SwinEncoder(nn.Module):
    def __init__(
        self,
        img_size=(256, 512),
        patch_size=4,
        in_chans=3,
        latent_dim=512,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 8),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
        final_channels=1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=0.0)

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()

        self.layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        num_features = embed_dim
        H, W = self.patch_embed.patches_resolution
        dp_idx = 0

        for i_layer, depth in enumerate(depths):
            blocks = nn.ModuleList()
            for i in range(depth):
                shift = (
                    (0, 0)
                    if (i % 2 == 0)
                    else (window_size[0] // 2, window_size[1] // 2)
                )
                blocks.append(
                    SwinBlock(
                        dim=num_features,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=dpr[dp_idx],
                    )
                )
                dp_idx += 1
            self.layers.append(blocks)

            if i_layer < len(depths) - 1:
                self.downsample_layers.append(PatchMerge((H, W), num_features))
                num_features *= 2
                H, W = H // 2, W // 2

        self.num_features = num_features
        self.norm = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.single_head = torch.nn.Linear(final_channels * num_features, latent_dim)
        # latent head MLP to smooth coupling into latent dynamics
        self.latent_head = MLP(
            in_dim=num_features, hidden=num_features, out_dim=num_features, p=drop
        )
        self.head = nn.Linear(num_features, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        H, W = self.patch_embed.patches_resolution

        for i, layer in enumerate(self.layers):
            for blk in layer:
                x = blk(x, H, W)
            if i < len(self.layers) - 1:
                x = self.downsample_layers[i](x)
                H, W = H // 2, W // 2

        x = self.norm(x)  # (B, L, C)
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.avgpool(x).flatten(1)  # (B, C)
        # x = self.latent_head(x)
        x = self.head(x)  # (B, latent_dim)

        # x = self.norm(x)
        # x = rearrange(x, "B L C -> B (C L)")
        # x = self.single_head(x)
        # x = self.norm2(x)
        return x


class SwinDecoder(nn.Module):
    def __init__(
        self,
        img_size=(256, 512),
        patch_size=4,
        out_chans=3,
        latent_dim=512,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 8),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        num_layers = len(depths)
        H_patch, W_patch = (img_size[0] // patch_size, img_size[1] // patch_size)
        dims = [embed_dim * (2**i) for i in range(num_layers)]
        resolutions = [
            (H_patch // (2**i), W_patch // (2**i)) for i in range(num_layers)
        ]

        final_dim = dims[-1]
        final_H, final_W = resolutions[-1]
        # latent tail MLP to smooth coupling back from latent vector
        self.latent_tail = MLP(
            in_dim=latent_dim, hidden=latent_dim, out_dim=latent_dim, p=drop
        )
        self.init_dense = nn.Linear(latent_dim, final_dim * final_H * final_W)
        self.init_H, self.init_W = final_H, final_W

        total_blocks = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        self.layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        for i_layer in range(num_layers - 1, -1, -1):
            dim = dims[i_layer]
            H, W = resolutions[i_layer]

            if i_layer < num_layers - 1:
                self.upsample_layers.append(PatchExpand((H // 2, W // 2), dim * 2))

            blocks = nn.ModuleList()
            depth = depths[i_layer]
            for i in range(depth):
                shift = (
                    (0, 0)
                    if (i % 2 == 0)
                    else (window_size[0] // 2, window_size[1] // 2)
                )
                blocks.append(
                    SwinBlock(
                        dim=dim,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=dpr[dp_idx],
                    )
                )
                dp_idx += 1
            self.layers.append(blocks)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.ConvTranspose2d(
            embed_dim, out_chans, kernel_size=patch_size, stride=patch_size
        )
        self.final_H = H_patch
        self.final_W = W_patch

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z = self.latent_tail(z)
        x = self.init_dense(z)
        x = rearrange(x, "b (h w c) -> b (h w) c", h=self.init_H, w=self.init_W)
        H, W = self.init_H, self.init_W

        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.upsample_layers[i - 1](x)
                H, W = H * 2, W * 2
            for blk in layer:
                x = blk(x, H, W)

        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.final_H, w=self.final_W)
        x = self.head(x)
        return x


# ------------------------------
# Autoencoder wrapper
# ------------------------------
class SwinAutoencoder(nn.Module):
    def __init__(
        self,
        img_size=(256, 512),
        patch_size=4,
        in_chans=3,
        latent_dim=512,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 8),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.encoder = SwinEncoder(
            img_size,
            patch_size,
            in_chans,
            latent_dim,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            drop,
            attn_drop,
            drop_path_rate,
        )
        self.decoder = SwinDecoder(
            img_size,
            patch_size,
            in_chans,
            latent_dim,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            drop,
            attn_drop,
            drop_path_rate,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


torch.set_default_dtype(torch.float64)
if __name__ == "__main__":
    # Expect the user to have: torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinAutoencoder(
        # img_size=(256, 512),
        img_size=(256, 768),
        patch_size=4,
        in_chans=3,
        latent_dim=256,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(8, 8),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
    ).to(device)

    x = torch.randn(4, 3, 256, 768, device=device)
    z = model.encode(x)
    print("Latent shape:", z.shape)
    xr = model.decode(z)
    print("Recon shape:", xr.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = nn.L1Loss()

    model.train()
    out = model(x)
    loss = criterion(out, x)
    loss.backward()
    opt.step()
    print(f"Loss: {loss.item():.4f}")
