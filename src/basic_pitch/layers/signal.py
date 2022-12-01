
from typing import Any, Callable, Optional
import jax.numpy as jnp
import jax.scipy.signal
from flax import linen as nn
from basic_pitch.layers.utils import LambdaLayer


class Stft(nn.Module):
    def setup(
        self,
        fft_length: int = 2048,
        hop_length: Optional[int] = None,
        window_length: Optional[int] = None,
        window_fn: Callable[[int, jnp.dtype], jnp.ndarray] = jnp.hanning,  # what is a hann window????
        pad_end: bool = False,
        center: bool = True,
        pad_mode: str = "REFLECT",
        name: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """
        STFT layer.
        The input is real-valued with shape (num_batches, num_samples).
        The output is complex-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            hop_length: The "stride" or number of samples to iterate before the start of the next frame.
            fft_length: FFT length.
            window_length: Window length. If None, then fft_length is used.
            window_fn: A callable that takes a window length and a dtype and returns a window.
            pad_end: Whether to pad the end of signals with zeros when the provided frame length and step produces
                a frame that lies partially past its end.
            center:
                If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
                If False, then D[:, t] begins at y[t * hop_length].
            pad_mode: Padding to use if center is True. One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
            name: Name of the layer.
            dtype: Type used in calcuation.
        """

        # super().setup()
        self.fft_length = fft_length
        self.window_length = window_length if window_length else self.fft_length
        self.hop_length = hop_length if hop_length else self.window_length // 4
        self.window_fn = window_fn
        self.final_window_fn = window_fn
        self.pad_end = pad_end
        self.center = center
        self.pad_mode = pad_mode

    def build(self, input_shape: jnp.shape) -> None:
        if self.window_length < self.fft_length:
            lpad = (self.fft_length - self.window_length) // 2
            rpad = self.fft_length - self.window_length - lpad

            def padded_window(window_length: int, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
                return jnp.pad(self.window_fn(window_length, dtype), (lpad, rpad))

            self.final_window_fn = padded_window

        if self.center:
            # not sure if these custom lambda layers work
            self.spec = LambdaLayer(
                lambda x: jnp.pad(
                    x,
                    [[0, 0] for _ in range(len(input_shape) - 1)] + [[self.fft_length // 2, self.fft_length // 2]],
                    mode=self.pad_mode,
                )
            )
        else:
            self.spec = LambdaLayer(lambda x: x)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.signal.stft(
            x=self.spec(inputs),
            nperseg=self.fft_length,
            # in tf, frame_step=number of samples to step=self.hop_length but what is the equivalent for scipy???
            noverlap=self.hop_length, # this is my guess
            nfft=self.fft_length,
            window=self.final_window_fn,
            padded=self.pad_end,
        )


class Spectrogram(Stft):
    def setup(
        self,
        power: int = 2,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A layer that calculates the magnitude spectrogram.
        The input is real-valued with shape (num_batches, num_samples).
        The output is real-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            power: Exponent to raise abs(stft) to.
            **kwargs: Any arguments that you'd pass to Stft
        """

        super().setup(*args, **kwargs)
        self.power = power

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(jnp.abs(super().__call__(inputs)), self.power)


class NormalizedLog(nn.Module):
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """

    def build(self, input_shape: jnp.shape) -> None:
        self.squeeze_batch = lambda batch: batch
        rank = len(input_shape)
        if rank == 4:
            assert input_shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
            self.squeeze_batch = lambda batch: jnp.squeeze(batch, axis=1)
        else:
            assert rank == 3, f"Only ranks 3 and 4 are supported!. Received rank {rank} for {input_shape}."

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs = self.squeeze_batch(inputs)
        power = jnp.square(inputs)
        log_power = 10 * jnp.log10(power + 1e-10)

        log_power_min = jnp.reshape(jnp.min(log_power, axis=(1, 2)), [jnp.shape(inputs)[0], 1, 1])
        log_power_offset = log_power - log_power_min
        log_power_offset_max = jnp.reshape(jnp.max(log_power_offset, axis=(1, 2)), [jnp.shape(inputs)[0], 1, 1])

        log_power_normalized = jnp.divide(log_power_offset, log_power_offset_max)  # need to account for nan maybe?

        return jnp.reshape(log_power_normalized, jnp.shape(inputs))
