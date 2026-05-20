import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset, DataLoader


def lorenz_derivatives(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Calculate derivatives for the Lorenz system at a given state.

    Parameters:
    -----------
    state : array_like
        State vector [x, y, z]
    sigma, rho, beta : float
        System parameters

    Returns:
    --------
    derivatives : array_like
        State derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return np.array([dx_dt, dy_dt, dz_dt])


def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    The Lorenz system of differential equations (scipy solver format).
    """
    return lorenz_derivatives(state, sigma, rho, beta)


def lorenz_soln(
    time_grid,
    initial_conditions=None,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    method="LSODA",
    rtol=1e-6,
    atol=1e-9,
    **kwargs,
):
    """
    Integrate the Lorenz system and return both derivatives and states as PyTorch tensors.

    Parameters:
    -----------
    time_grid : array_like
        Time points at which to evaluate the solution
    initial_conditions : array_like, optional
        Initial state vector [x0, y0, z0] (default: [1.0, 1.0, 1.0])
    sigma, rho, beta : float, optional
        System parameters with classic chaotic values as defaults
    method : str, optional
        Integration method (default: 'LSODA')
    rtol, atol : float, optional
        Relative and absolute tolerances for the solver
    **kwargs : dict, optional
        Additional keyword arguments passed to solve_ivp

    Returns:
    --------
    tuple: (deriv_state_vectors, state_vectors)
        - deriv_state_vectors: torch.Tensor of shape (len(time_grid), 3)
        - state_vectors: torch.Tensor of shape (len(time_grid), 3)
    """
    # Set default initial conditions if none provided
    if initial_conditions is None:
        initial_conditions = [2.0, 1.0, 1.0]

    time_grid = time_grid.cpu().numpy()  # Ensure time_grid is a numpy array

    # Create time span from the time grid
    t_span = (0.0, time_grid[-1])

    # Pack the system parameters
    params = (sigma, rho, beta)

    # Solve the system
    sol = solve_ivp(
        lorenz_system,
        t_span,
        initial_conditions,
        args=params,
        t_eval=time_grid,
        method=method,
        rtol=rtol,
        atol=atol,
        **kwargs,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Convert state solutions to shape (len(time_grid), 3)
    states_np = sol.y.T  # Shape: (len(time_grid), 3)

    # Calculate derivatives at each point
    derivatives_np = np.zeros_like(states_np)
    for i, state in enumerate(states_np):
        derivatives_np[i] = lorenz_derivatives(state, sigma, rho, beta)

    # Convert to PyTorch tensors
    states_torch = torch.tensor(states_np)
    derivatives_torch = torch.tensor(derivatives_np)

    return derivatives_torch, states_torch


class StaggeredODEDataset(Dataset):
    def __init__(
        self,
        delta_t,
        N_t,
        num_rollouts_per_sample,
        num_rollouts_to_integrate,
        ode_function,
        initial_conditions=None,
        warm_start_time=0.0,
        device=None,
    ):
        assert (
            num_rollouts_to_integrate % num_rollouts_per_sample == 0
        ), "num_rollouts_to_integrate must be divisible by num_rollouts_per_sample"

        # Compute the time grid:
        num_cells_total = num_rollouts_to_integrate * (N_t - 1)
        num_cell_boundaries_total = num_cells_total + 1
        self.time_grid = torch.linspace(
            warm_start_time,
            warm_start_time + delta_t * num_rollouts_to_integrate,
            num_cells_total + num_cell_boundaries_total,
        )

        self.J_hats = []
        self.u_hats = []

        # For each initial condition, integrate the ODE
        if initial_conditions is None:
            J_values, u_values = ode_function(self.time_grid, None)
            self.J_hats.append(J_values[:-1, :])
            self.u_hats.append(u_values[:-1, :])
        else:
            for initial_condition in initial_conditions:
                J_values, u_values = ode_function(
                    self.time_grid, initial_condition.cpu().numpy().tolist()
                )
                self.J_hats.append(J_values[:-1, :])
                self.u_hats.append(u_values[:-1, :])
        self.J_hats = torch.stack(self.J_hats)
        self.u_hats = torch.stack(self.u_hats)

        self.u_hats_mean, self.u_hats_std = self.u_hats.mean(
            dim=(0, 1), keepdim=True
        ), self.u_hats.std(dim=(0, 1), keepdim=True)

        self.u_hats = (self.u_hats - self.u_hats_mean) / self.u_hats_std
        self.J_hats = self.J_hats / self.u_hats_std

        self.t_scale = 1.0 / self.J_hats.std(dim=(0, 1, 2), keepdim=False)
        self.J_hats = self.t_scale * self.J_hats

        num_ics = self.J_hats.shape[0]
        self.num_time_points_per_sample = 2 * (N_t - 1) * num_rollouts_per_sample

        if device is not None:
            self.J_hats = self.J_hats.to(device)
            self.u_hats = self.u_hats.to(device)

        self.effective_length = (
            self.J_hats.shape[1] - self.num_time_points_per_sample
        ) * num_ics

    def __len__(self):
        return self.effective_length

    def __getitem__(self, idx):
        ic_ndx = idx // (self.J_hats.shape[1] - self.num_time_points_per_sample)
        t_ndx = idx % (self.J_hats.shape[1] - self.num_time_points_per_sample)

        J_hat = self.J_hats[ic_ndx, t_ndx : t_ndx + self.num_time_points_per_sample, :]
        u_hat = self.u_hats[ic_ndx, t_ndx : t_ndx + self.num_time_points_per_sample, :]
        J_init = J_hat[0, :]
        u_init = u_hat[0, :]
        J_hat = J_hat[0::2, :]
        u_hat = u_hat[1::2, :]
        return J_init, u_init, J_hat, u_hat


def staggered_ode_collate_fn(batch):
    """
    Custom collate function for the StaggeredODEDataset.

    Args:
        batch: List of tuples (J_init, u_init, J_hat, u_hat)

    Returns:
        Dictionary with keys 'J_init', 'u_init', 'J_hat', 'u_hat'
        containing batched tensors
    """
    J_inits, u_inits, J_hats, u_hats = zip(*batch)

    # Stack the tensors along a new batch dimension
    J_inits = torch.stack(J_inits)
    u_inits = torch.stack(u_inits)
    J_hats = torch.stack(J_hats)
    u_hats = torch.stack(u_hats)

    return {
        "J_init": J_inits,
        "u_init": u_inits,
        "J_hat": J_hats,
        "u_hat": u_hats,
    }


class StaggeredODEDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        **kwargs,
    ):
        # assert (
        #     len(dataset) % batch_size == 0
        # ), "Batch size must divide the dataset length evenly."

        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=staggered_ode_collate_fn,
            drop_last=True,
            **kwargs,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
