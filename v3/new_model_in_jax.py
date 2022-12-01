from typing import Union, List, Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import librosa
import pathlib


# basic outline:
# step 1: constant q stacking # Eran?
# step 2: harmonic stacking # Eran?

class Constants:
    #!/usr/bin/env python
    # encoding: utf-8
    #
    # Copyright 2022 Spotify AB
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    FFT_HOP = 256
    N_FFT = 8 * FFT_HOP

    NOTES_BINS_PER_SEMITONE = 1
    CONTOURS_BINS_PER_SEMITONE = 3
    # base frequency of the CENTRAL bin of the first semitone (i.e., the
    # second bin if annotations_bins_per_semitone is 3)
    ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
    ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
    AUDIO_SAMPLE_RATE = 22050
    AUDIO_N_CHANNELS = 1
    N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
    N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

    AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

    ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
    ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

    # ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
    ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

    # AUDIO_N_SAMPLES is the number of samples in the (clipped) audio that we use as input to the models
    AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

    DATASET_SAMPLING_FREQUENCY = {
        "MAESTRO": 5,
        "GuitarSet": 2,
        "MedleyDB-Pitch": 2,
        "iKala": 2,
        "slakh": 2,
    }


    def _freq_bins(bins_per_semitone: int, base_frequency: float, n_semitones: int) -> np.array:
        d = 2.0 ** (1.0 / (12 * bins_per_semitone))
        bin_freqs = base_frequency * d ** np.arange(bins_per_semitone * n_semitones)
        return bin_freqs


    FREQ_BINS_NOTES = _freq_bins(NOTES_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
    FREQ_BINS_CONTOURS = _freq_bins(CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)


class PosteriorgramModel(hk.Module):

    def __init__(self):
        # define blocks here:
        super().__init__()
        # self.cqt_harmonic_stacking = hk.Sequential([
        #     # cqt layer
        #     # harmonic stacking layer
        #     hk.IdentityCore()  # TODO: replace with actual signal processing
        # ])
        self.cqt_harmonic_stacking = hk.IdentityCore()
        
        self.top_branch = hk.Sequential([
            # 32-depth Conv2D (5x5), stride 1x3
            # batch norm
            # ReLU
            hk.Conv2D(32, (5, 5), stride=(1, 3)),
            hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
            jax.nn.relu
        ])
        self.yp_branch = hk.Sequential([
            # downward prong:
            # 16 x 5 x 5 Conv 2d
            # Batch Norm
            # ReLU
            # 8 x 3 x39 Conv 2D
            # Batch Norm, RELU
            # 1 Conv 2D, 5x5, sigmoid

            hk.Conv2D(16, (5, 5)),
            hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Conv2D(8, (3, 39)),
            hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Conv2D(1, (5, 5)),
            jax.nn.sigmoid
        ])
        self.yn_branch = hk.Sequential([
            # 32 conv 2d 7x7 stride 1x3
            # relu
            # 1 conv2d 7x3
            # sigmoid
            hk.Conv2D(32, (7, 7), stride=(1, 3)),
            jax.nn.relu,
            hk.Conv2D(1, (7, 3)),
            jax.nn.sigmoid
        ])
        self.yo_branch = hk.Sequential([
            # 1 conv2d 3x3
            # sigmoid
            hk.Conv2D(1, (3, 3)),
            jax.nn.sigmoid
        ])

    # define all model layers here as attributes

    def __call__(self, audio_path: Union[str, pathlib.Path]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * Constants.FFT_HOP
        hop_size = Constants.AUDIO_N_SAMPLES - overlap_len

        # audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)
        audio = librosa.load(audio_path, sr=11025)
        preprocessed = self.cqt_harmonic_stacking(audio)
        yp = self.yp_branch(preprocessed)
        yn = self.yn_branch(yp)
        top = self.top_branch(preprocessed)
        concat = jax.numpy.concatenate([top, yn])  # TODO: fix this probably
        yo = self.yo_branch(concat)
        print(yp, yn, yo)
        return yp, yn, yo


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

    audio_original, _ = librosa.load(str(audio_path), sr=Constants.AUDIO_SAMPLE_RATE, mono=True)

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
        frame(audio_original, Constants.AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (Constants.AUDIO_N_SAMPLES / Constants.AUDIO_SAMPLE_RATE),
        }
        for t_start in jnp.arange(audio_windowed.shape[0]) * hop_size / Constants.AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = jnp.abs(signal_length - frames_overlap) % jnp.abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = jnp.pad(signal, pad_axis, "constant", constant_values=pad_value)
    frames=signal.unfold(axis, frame_length, frame_step)
    return frames



def onset_posteriorgram_model(audio):  # Shalin, Mrunali
    pass
# def right_prong_up_to_concat():
# # step 3: 32-depth Conv2D (5x5), stride 1x3
    # step 4: batch norm
    # step 5: ReLU
   # output pass to concat?

# def concat(right_prong, notes):
    # concat
    # step 7: 1 Conv2d, 3x3
    # step 8: sigmoid activation?

# def pitch_model():
    # downward prong:
    # 16 x 5 x 5 Conv 2d
    # Batch Norm
    # ReLU
    # 8 x 3 x39 Conv 2D
    # Batch Norm, RELU
    # 1 Conv 2D, 5x5, sigmoid
# -> output,

# def notes_model(pitch_output):
    # 32 x 7x 7 COnv 2d
    # ReLU
    # 1 Conv 2D 7 x 3
    # sigmoid
    # return output
