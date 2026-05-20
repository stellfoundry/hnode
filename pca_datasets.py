import torch
from sklearn.utils.extmath import randomized_svd


def randomized_torch_svd(
    matrix, n_components=None, n_iter=4, random_state=None, target_device=None
):
    """
    Drop-in replacement for torch.linalg.svd using randomized SVD.

    Args:
        matrix: PyTorch tensor to decompose
        n_components: Number of components to compute (if None, computes min(matrix.shape))
        n_iter: Number of iterations for randomized algorithm (default 4)
        random_state: Random seed for reproducibility
        target_device: Device to return the results on (if None, uses matrix's device)

    Returns:
        U, S, Vh: Same format as torch.linalg.svd
    """
    # Convert to numpy and move to CPU if needed
    matrix_np = matrix.detach().cpu().numpy()

    # Default n_components to full rank if not specified
    if n_components is None:
        n_components = min(matrix_np.shape)

    # Compute randomized SVD
    U_np, S_np, Vh_np = randomized_svd(
        matrix_np,
        n_components=n_components,
        # n_oversamples=50,
        n_iter="auto",
        random_state=random_state,
        # svd_lapack_driver="gesvd",
    )

    # Convert back to PyTorch tensors on original device
    if target_device is None:
        device = matrix.device
    else:
        device = target_device
    dtype = matrix.dtype

    U = torch.from_numpy(U_np).to(device=device, dtype=dtype)
    S = torch.from_numpy(S_np).to(device=device, dtype=dtype)
    Vh = torch.from_numpy(Vh_np).to(device=device, dtype=dtype)

    return U, S, Vh


class PCADataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for PCA-based problems, e.g., flow past a cylinder.
    """

    def __init__(
        self,
        raw_data,
        z_data,
        dt,
        num_basis_functions=12,
        num_time_steps_per_sample=32,
        cache_label=None,
        cache_path=None,
        device="cpu",
        append_u_init=False,
        data_coefficients=None,
    ):
        """
        Initialize the dataset.

        Args:
            raw_data: Tensor containing the data to be processed. May be None
                if cache_path is provided.
            z_data: Tensor containing the conditioning data (e.g., Re number)
            dt: Time step size for the data
            num_basis_functions: Number of PCA basis functions to use
            num_time_steps_per_sample: Length of each time series sample
            cache_label: Optional label; looks for cache at /scratch/tmp/kinch/<label>.pt
            cache_path: Direct path to a cache .pt file containing keys
                'data_coefficients', 'basis_functions', and 'S'. Takes
                precedence over cache_label. Use this in preference to
                cache_label when distributing the dataset.
            device: Device to move tensors to
        """
        self.num_basis_functions = num_basis_functions
        self.num_time_steps_per_sample = num_time_steps_per_sample
        self.device = device
        self.append_u_init = append_u_init
        self.dt = dt

        self._process_data(
            raw_data, z_data, dt, cache_label, cache_path, data_coefficients
        )

    def _process_data(
        self, raw_data, z_data, dt, cache_label, cache_path, data_coefficients
    ):
        if data_coefficients is None:
            # Try to load data_coefficients, basis_functions, and S from a cache file.
            # cache_path (a direct file path) takes precedence over cache_label.
            try:
                if cache_path is not None:
                    flnm = cache_path
                elif cache_label is not None:
                    flnm = f"/scratch/tmp/kinch/{cache_label}.pt"
                else:
                    flnm = None
                    raise FileNotFoundError("No cache_path or cache_label provided.")
                cache = torch.load(flnm, map_location=self.device)
                self.data_coefficients = cache["data_coefficients"]
                self.basis_functions = cache["basis_functions"]
                self.S = cache["S"]
                if self.data_coefficients.shape[1] != self.num_basis_functions:
                    raise ValueError(f"Mismatch in number of basis functions in cache")
            except (FileNotFoundError, KeyError, ValueError):
                # Fall back to computing SVD from raw_data
                data_matrix = raw_data.reshape(-1, *raw_data.shape[2:])
                data_matrix = data_matrix.reshape(data_matrix.shape[0], -1).T
                data_matrix_clean = data_matrix.to(self.device)
                print(f"Data matrix shape after cleaning: {data_matrix_clean.shape}")
                U, S, Vh = randomized_torch_svd(
                    data_matrix_clean,
                    n_components=self.num_basis_functions,
                    n_iter="auto",
                    random_state=42,
                    target_device=self.device,
                )
                self.basis_functions = U.T
                self.S = S
                self.data_coefficients = (torch.diag(S) @ Vh).T
                # Save to file for future use
                if flnm is not None:
                    torch.save(
                        {
                            "data_coefficients": self.data_coefficients,
                            "basis_functions": self.basis_functions,
                            "S": self.S,
                        },
                        flnm,
                    )
        else:
            self.data_coefficients = data_coefficients

        self.z_data = z_data.to(self.device).to(torch.float64)
        self.num_zs = self.z_data.shape[0]

        self.data_coefficients = self.data_coefficients.reshape(
            z_data.shape[0], -1, self.num_basis_functions
        )

        self.J_hats = (
            self.data_coefficients[:, 1:, :] - self.data_coefficients[:, :-1, :]
        ) / dt
        self.u_hats = self.data_coefficients

        self.J_hats = self.J_hats.to(torch.float64)
        self.u_hats = self.u_hats.to(torch.float64)

        self.u_hats_mean, self.u_hats_std = self.u_hats.mean(
            dim=(0, 1, 2), keepdim=True
        ), self.u_hats.std(dim=(0, 1, 2), keepdim=True)

        self.u_hats = (self.u_hats - self.u_hats_mean) / self.u_hats_std
        self.J_hats = self.J_hats / self.u_hats_std

        self.acceleration = (self.J_hats[:, 1:, :] - self.J_hats[:, :-1, :]) / dt

        self.t_scale = 1.0 / self.J_hats.std(keepdim=False)
        self.J_hats = self.t_scale * self.J_hats

        # Calculate the effective length, taking into account the number of time steps needed for a full sample
        self.num_time_steps = self.u_hats.shape[1]
        self.effective_length = (
            self.num_time_steps - self.num_time_steps_per_sample - 1
        ) * self.num_zs

    def __len__(self):
        return self.effective_length

    def get_item_by_ndx(self, z_ndx, t_ndx):
        sample_u_hats = self.u_hats[
            z_ndx, t_ndx : t_ndx + self.num_time_steps_per_sample + 1, :
        ]
        sample_J_hats = self.J_hats[
            z_ndx, t_ndx : t_ndx + self.num_time_steps_per_sample, :
        ]

        J_hat = sample_J_hats
        J_init = sample_J_hats[0, :]
        u_hat = sample_u_hats[1:, :]
        u_init = (sample_u_hats[0, :] + sample_u_hats[1, :]) / 2.0

        if self.append_u_init:
            z = torch.cat([self.z_data[z_ndx], self.u_hats[z_ndx, 0]], dim=-1)
        else:
            z = self.z_data[z_ndx]

        # Create batch dictionary
        batch = {
            "J_init": J_init,
            "u_init": u_init,
            "J_hat": J_hat,
            "u_hat": u_hat,
            "z": z,
            "t_ndx": torch.tensor(t_ndx, device=self.device, dtype=torch.long),
            "z_ndx": torch.tensor(z_ndx, device=self.device, dtype=torch.long),
            "sample_u_hats": sample_u_hats,
            "sample_J_hats": sample_J_hats,
        }

        return batch

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Starting time index for the sample

        Returns:
            dict: Batch dictionary with keys 'J_init', 'u_init', 'u_hat'
        """
        if idx >= self.effective_length:
            raise IndexError(
                f"Index {idx} is out of range. Max index: {self.effective_length - 1}"
            )

        z_ndx = idx // (self.effective_length // self.num_zs)
        t_ndx = idx % (self.effective_length // self.num_zs)

        return self.get_item_by_ndx(z_ndx, t_ndx)


def create_dataloader(
    dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True, **kwargs
):
    """
    Create a DataLoader for a PCADataset class object.

    Args:
        dataset: Instance of a PCADataset class
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        drop_last: Whether to drop the last incomplete batch
        **kwargs: Additional arguments for DataLoader

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader
    """

    def collate_fn(batch_list):
        """Custom collate function to properly batch the dictionary outputs."""
        batched = {}
        for key in batch_list[0].keys():
            batched[key] = torch.stack([item[key] for item in batch_list])
        return batched

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **kwargs,
    )
