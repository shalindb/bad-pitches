import jax.numpy as jnp
from typing import Any, List, Optional, Tuple, Union
import haiku as hk
import scipy.signal
import warnings
from jax import lax

def create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = 256,
    transition_bandwidth: float = 0.03,
) -> jnp.ndarray:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through. Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is
    the Nyquist frequency of the signal BEFORE downsampling.
    """

    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passband_max] is the frequency range we want to keep
    # untouched and [stopband_min, 1.0] is the range we want to remove
    key_frequencies = [0.0, passband_max, stopband_min, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gain_at_key_frequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filter_kernel = scipy.signal.firwin2(kernel_length, key_frequencies, gain_at_key_frequencies)

    return jnp.array(filter_kernel)

def next_power_of_2(A: int) -> int:
    """A helper function to calculate the next nearest number to the power of 2."""
    return int(jnp.ceil(jnp.log2(A)))

def early_downsample(
    sr: Union[float, int],
    hop_length: int,
    n_octaves: int,
    nyquist_hz: float,
    filter_cutoff_hz: float,
) -> Tuple[Union[float, int], int, int]:
    """Return new sampling rate and hop length after early downsampling"""
    downsample_count = early_downsample_count(nyquist_hz, filter_cutoff_hz, hop_length, n_octaves)
    downsample_factor = 2 ** (downsample_count)

    hop_length //= downsample_factor  # Getting new hop_length
    new_sr = sr / float(downsample_factor)  # Getting new sampling rate

    return new_sr, hop_length, downsample_factor


# The following two downsampling count functions are obtained from librosa CQT
# They are used to determine the number of pre resamplings if the starting and ending frequency
# are both in low frequency regions.
def early_downsample_count(nyquist_hz: float, filter_cutoff_hz: float, hop_length: int, n_octaves: int) -> int:
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(0, int(jnp.ceil(jnp.log2(0.85 * nyquist_hz / filter_cutoff_hz)) - 1) - 1)
    num_twos = next_power_of_2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def get_early_downsample_params(
    sr: Union[float, int], hop_length: int, fmax_t: float, Q: float, n_octaves: int
) -> Tuple[Union[float, int], int, float, jnp.ndarray, bool]:
    """Compute downsampling parameters used for early downsampling"""

    window_bandwidth = 1.5  # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    sr, hop_length, downsample_factor = early_downsample(sr, hop_length, n_octaves, sr // 2, filter_cutoff)
    if downsample_factor != 1:
        earlydownsample = True
        early_downsample_filter = create_lowpass_filter(
            band_center=1 / downsample_factor,
            kernel_length=256,
            transition_bandwidth=0.03,
        )
    else:
        early_downsample_filter = None
        earlydownsample = False

    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample


def get_window_dispatch(window: Union[str, Tuple[str, float]], N: int, fftbins: bool = True) -> jnp.ndarray:
    if isinstance(window, str):
        return scipy.signal.get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = jnp.floor(-N / 2 / jnp.sqrt(-2 * jnp.log(10 ** (-window[1] / 20))))
            return scipy.signal.get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True,
) -> Tuple[jnp.ndarray, int, jnp.ndarray, jnp.ndarray]:
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** next_power_of_2(jnp.ceil(Q * fs / fmin))

    if (fmax is not None) and (n_bins is None):
        n_bins = jnp.ceil(bins_per_octave * jnp.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = jnp.ceil(bins_per_octave * jnp.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (jnp.r_[0:n_bins] / float(bins_per_octave))

    if jnp.max(freqs) > fs / 2 and topbin_check is True:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(jnp.max(freqs))
        )

    tempKernel = jnp.zeros((int(n_bins), int(fftLen)), dtype=jnp.complex64)

    lengths = jnp.ceil(Q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        _l = jnp.ceil(Q * fs / freq)

        # Centering the kernels, pad more zeros on RHS
        start = int(jnp.ceil(fftLen / 2.0 - _l / 2.0)) - int(_l % 2)

        sig = (
            get_window_dispatch(window, int(_l), fftbins=True)
            * jnp.exp(jnp.r_[-_l // 2 : _l // 2] * 1j * 2 * jnp.pi * freq / fs)
            / _l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel = tempKernel.at[k, start : start + int(_l)].set(sig / jnp.linalg.norm(sig, norm))
        else:
            tempKernel = tempKernel.at[k, start : start + int(_l)].set(sig)

    return tempKernel, fftLen, lengths, freqs


def get_cqt_complex(
    x: jnp.ndarray,
    cqt_kernels_real: jnp.ndarray,
    cqt_kernels_imag: jnp.ndarray,
    hop_length: int,
    padding: hk.Module,
) -> jnp.ndarray:
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    try:
        x = padding(x)  # When center is True, we need padding at the beginning and ending
    except Exception:
        warnings.warn(
            f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
            "padding with reflection mode might not be the best choice, try using constant padding",
            UserWarning,
        )
        x = jnp.pad(x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2))
    CQT_real = jnp.transpose(
        lax.conv(
            jnp.transpose(x, [0, 2, 1]),
            jnp.transpose(cqt_kernels_real, [2, 1, 0]),
            padding="VALID",
            window_strides=hop_length,
        ),
        [0, 2, 1],
    )
    CQT_imag = -jnp.transpose(
        lax.conv(
            jnp.transpose(x, [0, 2, 1]),
            jnp.transpose(cqt_kernels_imag, [2, 1, 0]),
            padding="VALID",
            window_strides=hop_length,
        ),
        [0, 2, 1],
    )

    return jnp.stack((CQT_real, CQT_imag), axis=-1)


def downsampling_by_n(x: jnp.ndarray, filter_kernel: jnp.ndarray, n: float, match_torch_exactly: bool = True) -> jnp.ndarray:
    """
    Downsample the given tensor using the given filter kernel.
    The input tensor is expected to have shape `(n_batches, channels, width)`,
    and the filter kernel is expected to have shape `(num_output_channels,)` (i.e.: 1D)

    If match_torch_exactly is passed, we manually pad the input rather than having TensorFlow do so with "SAME".
    The result is subtly different than Torch's output, but it is compatible with TensorFlow Lite (as of v2.4.1).
    """

    if match_torch_exactly:
        paddings = [
            [0, 0],
            [0, 0],
            [(filter_kernel.shape[-1] - 1) // 2, (filter_kernel.shape[-1] - 1) // 2],
        ]
        padded = jnp.pad(x, paddings)

        # Store this tensor in the shape `(n_batches, width, channels)`
        padded_nwc = jnp.transpose(padded, [0, 2, 1])
        result_nwc = lax.conv(padded_nwc, filter_kernel[:, None, None], padding="VALID", window_strides=n)
    else:
        x_nwc = jnp.transpose(x, [0, 2, 1])
        result_nwc = lax.conv(x_nwc, filter_kernel[:, None, None], padding="SAME", window_strides=n)
    result_ncw = jnp.transpose(result_nwc, [0, 2, 1])
    return result_ncw


class ReflectionPad1D(hk.Module):
    """
    Replica of Torch's nn.ReflectionPad1D in haiku.
    """

    def __init__(self, padding: Union[int, Tuple[int]] = 1):
        super().__init__()
        self.padding = padding

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.pad(x, [[0, 0], [0, 0], [self.padding, self.padding]], "REFLECT")


class ConstantPad1D(hk.Module):
    """
    Replica of Torch's nn.ConstantPad1D in haiku.
    """

    def __init__(self, padding: Union[int, Tuple[int]] = 1, value: int = 0):
        super().__init__()
        self.padding = padding
        self.value = value

    def call(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.pad(x, [[0, 0], [0, 0], [self.padding, self.padding]], "CONSTANT", self.value)


class CQTLayer(hk.Module):
    def __init__(
        self,
        input_shape,
        sample_rate: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.70,
        fmax: Optional[float] = None,
        n_bins: int = 84,
        filter_scale: int = 1,
        bins_per_octave: int = 12,
        norm: bool = True,
        basis_norm: int = 1,
        window: str = "hann",
        pad_mode: str = "reflect",
        earlydownsample: bool = True,
        trainable: bool = False,
        output_format: str = "Magnitude",
        match_torch_exactly: bool = True,
    ):
        super().__init__()
        self.sample_rate: Union[float, int] = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_bins = n_bins
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.norm = norm
        self.basis_norm = basis_norm
        self.window = window
        self.pad_mode = pad_mode
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        self.match_torch_exactly = match_torch_exactly
        self.normalization_type = "librosa"

        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(self.filter_scale) / (2 ** (1 / self.bins_per_octave) - 1)

        self.lowpass_filter = create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.001)

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(self.bins_per_octave, self.n_bins)
        self.n_octaves = int(jnp.ceil(float(self.n_bins) / self.bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = self.fmin * 2 ** (self.n_octaves - 1)
        remainder = self.n_bins % self.bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((self.bins_per_octave - 1) / self.bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / self.bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (1 - 1 / self.bins_per_octave)  # Adjusting the top minium bins
        if fmax_t > self.sample_rate / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(fmax_t)
            )

        if self.earlydownsample is True:  # Do early downsampling if this argument is True
            (
                self.sample_rate,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(self.sample_rate, self.hop_length, fmax_t, Q, self.n_octaves)

            self.early_downsample_filter = early_downsample_filter
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            self.sample_rate,
            self.fmin_t,
            n_filters,
            self.bins_per_octave,
            norm=self.basis_norm,
            topbin_check=False,
        )

        # For the normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = self.fmin * 2.0 ** (jnp.r_[0 : self.n_bins] / float(self.bins_per_octave))
        self.frequencies = freqs

        self.lengths = jnp.ceil(Q * self.sample_rate / freqs)

        self.basis = basis
        # NOTE(psobot): this is where the implementation here starts to differ from CQT2010.

        # These cqt_kernel is already in the frequency domain
        self.cqt_kernels_real = jnp.expand_dims(basis.real, 1)
        self.cqt_kernels_imag = jnp.expand_dims(basis.imag, 1)

        if self.trainable:
            self.cqt_kernels_real = hk.get_parameter("cqt_kernels_real", self.cqt_kernels_real.shape, init=self.cqt_kernels_real)
            self.cqt_kernels_imag = hk.get_parameter("cqt_kernels_imag", self.cqt_kernels_imag.shape, init=self.cqt_kernels_imag)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = ConstantPad1D(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = ReflectionPad1D(self.n_fft // 2)

        rank = len(input_shape)
        if rank == 2:
            self.reshape_input = lambda x: x[:, None, :]
        elif rank == 1:
            self.reshape_input = lambda x: x[None, None, :]
        elif rank == 3:
            self.reshape_input = lambda x: x
        else:
            raise ValueError(f"Input shape must be rank <= 3, found shape {input_shape}")
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.reshape_input(x)  # type: ignore

        if self.earlydownsample is True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor, self.match_torch_exactly)

        hop = self.hop_length

        # Getting the top octave CQT
        CQT = get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)

        x_down = x  # Preparing a new variable for downsampling

        for _ in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_n(x_down, self.lowpass_filter, 2, self.match_torch_exactly)
            CQT1 = get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
            CQT = jnp.concatenate((CQT1, CQT), axis=1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins

        # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it
        # same mag as 1992
        CQT = CQT * self.downsample_factor

        # Normalize again to get same result as librosa
        if self.normalization_type == "librosa":
            CQT *= jnp.sqrt(self.lengths.reshape((-1, 1, 1)))
        elif self.normalization_type == "convolutional":
            pass
        elif self.normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % self.normalization_type)

        # Transpose the output to match the output of the other spectrogram layers.
        if self.output_format.lower() == "magnitude":
            # Getting CQT Amplitude
            return jnp.transpose(jnp.sqrt(jnp.sum(jnp.square(CQT), axis=-1)), [0, 2, 1])

        elif self.output_format.lower() == "complex":
            return CQT

        elif self.output_format.lower() == "phase":
            phase_real = jnp.cos(jnp.arctan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = jnp.sin(jnp.arctan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return jnp.stack((phase_real, phase_imag), axis=-1)
        

# Harmonic Stacking
class HarmonicStackingLayer(hk.Module):
    def __init__(
        self, bins_per_semitone: int, harmonics: List[float], n_output_freqs: int, name: str = "harmonic_stacking"
    ):
        """Downsample frequency by stride, upsample channels by 4."""
        super().__init__()
        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.shifts = [
            int(jnp.round(12.0 * self.bins_per_semitone * jnp.log2(float(h)))) for h in self.harmonics
        ]
        self.n_output_freqs = n_output_freqs
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # (n_batch, n_times, n_freqs, 1)
        # tf.debugging.assert_equal(tf.shape(x).shape, 4)
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


