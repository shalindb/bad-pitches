import jax.numpy as jnp
from flax import linen as nn

from typing import Any, List


class HarmonicStacking(nn.Module):
    """Harmonic stacking layer

    Input shape: (n_batch, n_times, n_freqs, 1)
    Output shape: (n_batch, n_times, n_output_freqs, len(harmonics))

    n_freqs should be much larger than n_output_freqs so that information from the upper
    harmonics is captured.

    Attributes:
        bins_per_semitone: The number of bins per semitone of the input CQT
        harmonics: List of harmonics to use. Should be positive numbers.
        shifts: A list containing the number of bins to shift in frequency for each harmonic
        n_output_freqs: The number of frequency bins in each harmonic layer.
    """

    def setup(
        self, bins_per_semitone: int, harmonics: List[float], n_output_freqs: int, name: str = "harmonic_stacking"
    ):
        """Downsample frequency by stride, upsample channels by 4."""
        # super().setup()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [int(jnp.round(12.0 * self.bins_per_semitone * jnp.log2(float(h))) for h in self.harmonics)]
        self.n_output_freqs = n_output_freqs

    def get_config(self) -> Any:
        config = {}
        if (super().get_config):
            config = super().get_config().copy()
        config.update(
            {
                "bins_per_semitone": self.bins_per_semitone,
                "harmonics": self.harmonics,
                "n_output_freqs": self.n_output_freqs,
                "name": self.name,
            }
        )
        return config

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        assert (len(x.shape) == 4)
        channels = []
        for shift in self.shifts:
            if shift == 0:
                padded = x
            elif shift > 0:
                paddings = jnp.array([[0, 0], [0, 0], [0, shift], [0, 0]])
                padded = jnp.pad(x[:, :, shift:, :], paddings)
            elif shift < 0:
                paddings = jnp.array([[0, 0], [0, 0], [-shift, 0], [0, 0]])
                padded = jnp.pad(x[:, :, :shift, :], paddings)
            else:
                raise ValueError

            channels.append(padded)
        x = jnp.concatenate(channels, axis=-1)
        x = x[:, :, : self.n_output_freqs, :]  # return only the first n_output_freqs frequency channels
        return x


class FlattenAudioCh(nn.Module):
    """Layer which removes a "channels" dimension of size 1.

    Input shape: (batch, time, 1)
    Output shape: (batch, time)
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, time, ch)"""
        assert (x.shape[2] == 1)
        return x.reshape(x.shape[0], x.shape[1])


class FlattenFreqCh(nn.Module):
    """Layer to flatten the frequency channel and make each channel
    part of the frequency dimension.

    Input shape: (batch, time, freq, ch)
    Output shape: (batch, time, freq*ch)
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (batch, time, freq, ch)"""
        return x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
