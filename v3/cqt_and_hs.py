from IPython.lib.display import Audio
import jax.numpy as jnp
import librosa
from constants import *
from typing import Union, List, Dict, Tuple
import pathlib
import jax
import numpy as np 

harmonics = jnp.array([0.5] + list(range(1, 8)))
def harmonic_stacking(x, bins_per_semitone=CONTOURS_BINS_PER_SEMITONE, harmonics=harmonics, n_output_freqs=N_FREQ_BINS_CONTOURS):
    """Downsample frequency by stride, upsample channels by 4."""
    shifts = [int(jnp.round(12.0 * bins_per_semitone * jnp.log2(float(h)))) for h in harmonics]
    channels = []
    for shift in shifts:
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
    x = x[:, :, : n_output_freqs, :]  # return only the first n_output_freqs frequency channels
    return x

def cqt_windowed(audio_array, sr=22050, hop_length=256, bins_per_octave=12*CONTOURS_BINS_PER_SEMITONE, n_bins=N_FREQ_BINS_CONTOURS):
    # audio = audio.reshape(1, audio.shape[0])
    cqt = librosa.cqt(audio_array, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=ANNOTATIONS_BASE_FREQUENCY)
    # reshape now:
    cqt = cqt.reshape(1, *cqt.shape)
    cqt = jnp.abs(cqt)
    return jnp.expand_dims(jnp.array(cqt), axis=-1)

def load_and_cqt(audio_path, sr=22050, hop_length=512, bins_per_octave=12, n_bins=84):
    audio, _ = librosa.load(audio_path, sr=sr)
    # # audio = audio.reshape(1, audio.shape[0])
    # cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
    # print(cqt.shape)
    # # reshape now:
    # cqt = cqt.reshape(1, *cqt.shape)
    # cqt = jnp.abs(cqt)
    # return jnp.expand_dims(jnp.array(cqt), axis=-1)
    return cqt_windowed(audio, sr, hop_length, bins_per_octave, n_bins)

# Returns (cqt'd audio tensor, window_num) with a random window num 
def load_cqt_window(path, sr=AUDIO_SAMPLE_RATE, hop_length=256):
    duration = librosa.get_duration(filename=path)
    window_num = np.random.randint((duration // AUDIO_WINDOW_LENGTH) - AUDIO_WINDOW_LENGTH)
    raw_audio = librosa.load(path, sr=sr, offset=window_num * AUDIO_WINDOW_LENGTH, duration=AUDIO_WINDOW_LENGTH)[0]
    return (cqt_windowed(raw_audio, sr, hop_length), window_num)

def get_audio_input(
    audio_path: Union[pathlib.Path, str], overlap_len: int, hop_size: int
) -> Tuple[jnp.ndarray, List[Dict[str, int]], int]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: ndarray with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = jnp.concatenate([jnp.zeros((int(overlap_len / 2),), dtype=jnp.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length

def window_audio_file(audio_original: jnp.ndarray, hop_size: int) -> Tuple[jnp.ndarray, List[Dict[str, int]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: ndarray with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """

    audio_windowed = jnp.expand_dims(
        frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in jnp.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    print(signal.shape)
    print("frame_length:", frame_length)
    print("frame_step: ", frame_step)
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = jnp.abs(signal_length - frames_overlap) % jnp.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        print(signal.ndim)
        if pad_size != 0:
            pad_axis = [(0,0)] * signal.ndim
            print(pad_axis)
            pad_axis[axis] = (0, pad_size)

            signal = jnp.pad(signal, pad_axis, "constant", constant_values=pad_value)
    
    signal = jnp.expand_dims(signal, axis=-1)
    signal = jnp.expand_dims(signal, axis=0)
    print("signal shape: ", signal.shape)
    # new pad:
    
    
    
    new_shape = (1, (signal.shape[1] - frame_length) // frame_step + 1, frame_length, 1)
    new_arr = np.zeros(shape = new_shape)
    print(new_arr.shape)
    window_starts = range(0, signal.shape[1] - frame_length, frame_step)
    print(len(window_starts))
    for i, w_i_start in enumerate(window_starts):
      new_arr[:, i, :, :] = signal[:,w_i_start: w_i_start + frame_length,:]
    
    return jnp.array(new_arr)
    # return signal
    #frames=signal.reshape(_,_,_,_,1)
    #signal.unfold(axis, frame_length, frame_step)