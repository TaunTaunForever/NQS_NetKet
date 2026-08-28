from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PaperMLPLogPsi(nn.Module):
    """Single-state version of the paper's TFI MLP target network."""

    hidden_width: int = 128
    num_layers: int = 4

    @nn.compact
    def __call__(self, sigma):
        x = jnp.asarray(sigma, dtype=jnp.float64)
        for layer in range(self.num_layers):
            x = nn.Dense(self.hidden_width, name=f"dense_{layer}")(x)
            x = nn.LayerNorm(name=f"layer_norm_{layer}")(x)
            x = nn.gelu(x)
        out = nn.Dense(2, name="logpsi_out")(x)
        return out[..., 0] + 1j * out[..., 1]


class _CircularConv(nn.Module):
    features: int
    kernel_size: tuple[int, int] = (3, 3)

    @nn.compact
    def __call__(self, x):
        pad_x = self.kernel_size[0] // 2
        pad_y = self.kernel_size[1] // 2
        x = jnp.pad(x, ((0, 0), (pad_x, pad_x), (pad_y, pad_y), (0, 0)), mode="wrap")
        return nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            padding="VALID",
        )(x)


class _ResidualBlock(nn.Module):
    channels: int = 16

    @nn.compact
    def __call__(self, x):
        y = _CircularConv(self.channels, name="conv_0")(x)
        y = nn.LayerNorm(name="layer_norm_0")(y)
        y = nn.gelu(y)
        y = _CircularConv(self.channels, name="conv_1")(y)
        y = nn.LayerNorm(name="layer_norm_1")(y)
        y = nn.gelu(y)
        return x + y


class PaperSquareJ1J2ResNetLogPsi(nn.Module):
    """Residual CNN target described for the paper's square J1-J2 benchmark."""

    length: int
    channels: int = 16
    num_blocks: int = 4
    use_d4_symmetry_average: bool = True
    enforce_spin_inversion: bool = True
    penalize_nonzero_magnetization: bool = True

    def _d4_stack(self, x):
        transforms = [
            x,
            jnp.rot90(x, 1, axes=(1, 2)),
            jnp.rot90(x, 2, axes=(1, 2)),
            jnp.rot90(x, 3, axes=(1, 2)),
            jnp.flip(x, axis=1),
            jnp.flip(x, axis=2),
            jnp.swapaxes(x, 1, 2),
            jnp.flip(jnp.swapaxes(x, 1, 2), axis=1),
        ]
        return jnp.stack(transforms, axis=1)

    @nn.compact
    def __call__(self, sigma):
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        magnetization = jnp.sum(sigma, axis=-1)
        if self.enforce_spin_inversion:
            sigma = jnp.where(sigma[:, :1] < 0, -sigma, sigma)

        x = sigma.reshape((sigma.shape[0], self.length, self.length))
        if self.use_d4_symmetry_average:
            x = self._d4_stack(x)
        else:
            x = x[:, None, :, :]

        batch_size, n_sym, length_x, length_y = x.shape
        x = x.reshape((batch_size * n_sym, length_x, length_y, 1))
        x = _CircularConv(self.channels, name="input_conv")(x)
        x = nn.LayerNorm(name="input_layer_norm")(x)
        x = nn.gelu(x)

        for block in range(self.num_blocks):
            x = _ResidualBlock(self.channels, name=f"residual_block_{block}")(x)

        x = jnp.mean(x, axis=(1, 2))
        out = nn.Dense(2, name="logpsi_out")(x)
        logpsi = out[..., 0] + 1j * out[..., 1]
        logpsi = logpsi.reshape((batch_size, n_sym))
        logpsi = jnp.mean(logpsi, axis=1)

        if self.penalize_nonzero_magnetization:
            penalty = jnp.where(jnp.abs(magnetization) > 0, 30.0, 0.0)
            logpsi = logpsi - penalty
        return logpsi
