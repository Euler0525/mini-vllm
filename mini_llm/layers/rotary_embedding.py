import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    r"""
    Apply a 90° rotation to the two halves of the last dimension.

    For x = [x1, x2],

        rotate_half(x) = [-x2, x1].

    This gives the RoPE identity

        R(theta)x
        = x cos(theta) + rotate_half(x) sin(theta).
    """
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    r"""
    Rotary position embedding for Q/K shaped

        (..., seq_len, num_heads, head_dim).

    For head dimension d, define inverse frequencies

        omega_i = base^(-2i / d),
        i = 0, ..., d/2 - 1.

    At position m, the rotary phase is

        theta_{m,i} = m * omega_i.

    RoPE rotates Q and K as

        q'_m = R(theta_m) q_m,
        k'_m = R(theta_m) k_m,

    with

        R(theta)x
        = x cos(theta) + rotate_half(x) sin(theta).

    The resulting Q-K dot product depends on relative position:

        (q'_m)^T k'_n
        = q_m^T R(theta_n - theta_m) k_n.
    """

    def __init__(self, head_dim: int, base: float):
        super().__init__()

        # omega_i = base^(-2i / head_dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape
                (..., seq_len, num_heads, head_dim).
            k: Key tensor of shape
                (..., seq_len, num_heads, head_dim).
            positions: Position indices of shape (..., seq_len).

        Returns:
            Rotated query and key tensors with the same shapes as q and k.

        For each position m,

            theta_m = m * inv_freq,

            RoPE(x_m)
            = x_m cos(theta_m)
              + rotate_half(x_m) sin(theta_m).
        """
        # theta_{m,i} = m * omega_i
        freqs = (
            positions.to(device=q.device, dtype=torch.float32)[..., None]
            * self.inv_freq
        )

        # Match the half-split layout used by rotate_half:
        # [theta_0, ..., theta_n] -> [theta_0, ..., theta_n,
        #                             theta_0, ..., theta_n]
        angles = torch.cat((freqs, freqs), dim=-1)

        # (..., seq_len, 1, head_dim), broadcast over attention heads.
        cos = angles.cos().to(q.dtype).unsqueeze(-2)
        sin = angles.sin().to(q.dtype).unsqueeze(-2)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        return q, k
