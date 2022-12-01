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

from typing import Any, Callable, Dict
# import numpy as np

import jax
import jax.numpy as jnp
from flax import linen
import optax


from basic_pitch import nn #as bpnn; TODO: change to bpnn
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch.layers import signal, nnaudio

jxln = linen # TODO: change jxln back to nn. 
# tfkl = tf.keras.layers
# tfkl = nn.Module

MAX_N_SEMITONES = int(jnp.floor(12.0 * jnp.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))


def smoothed_bce_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, label_smoothing: float) -> jnp.ndarray:
    """Replication of keras.losses.binary_cross_entropy() in jax, with optax
    Args:
        y_true: the true labels
        y_pred: the predicted labels
        label_smoothing: Alpha value to smooth values
    
    Returns: 
        The smoothened transcription loss. 
    
    """
    # may need to check whether logits or labels themselves passed here?
    # also, dimensionality, # of classes, et.c 
    if label_smoothing > 0:
        y_pred = optax.smooth_labels(y_pred, label_smoothing)

    bce = optax.softmax_cross_entropy(y_pred, y_true)
    return bce


def transcription_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, label_smoothing: float) -> jnp.ndarray:
    """Really a binary cross entropy loss. Used to calculate the loss between the predicted
    posteriorgrams and the ground truth matrices.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Squeeze labels towards 0.5 or some other value. 

    Returns:
        The transcription loss.
    """
    bce = smoothed_bce_loss(y_true=y_true, y_pred=y_pred, label_smoothing=label_smoothing)
    return bce


def weighted_transcription_loss(
    y_true: jnp.ndarray, y_pred: jnp.ndarray, label_smoothing: float, positive_weight: float = 0.5
) -> jnp.ndarray:
    """The transcription loss where the positive and negative true labels are balanced by a weighting factor.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        The weighted transcription loss.
    """
    negative_mask = jnp.equal(y_true, 0)
    nonnegative_mask = jnp.logical_not(negative_mask)
    

    # TODO: not sure if converted correctly, may have to check . 
    bce_negative = smoothed_bce_loss(
        jnp.where(negative_mask, y_true, 0),
        jnp.where(negative_mask, y_pred, 1), # 1 allows for loss contribution to be 0?
        label_smoothing=label_smoothing,
    )
    bce_nonnegative = smoothed_bce_loss(
        jnp.where(nonnegative_mask, y_true, 0),
        jnp.where(nonnegative_mask, y_pred, 0),
        label_smoothing=label_smoothing,
    )
    return ((1 - positive_weight) * bce_negative) + (positive_weight * bce_nonnegative)


def onset_loss(
    weighted: bool, label_smoothing: float, positive_weight: float
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """

    Args:
        weighted: Whether or not to use a weighted cross entropy loss.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A function that calculates the transcription loss. The function will
        return weighted_transcription_loss if weighted is true else it will return
        transcription_loss.
    """
    if weighted:
        return lambda x, y: weighted_transcription_loss(
            x, y, label_smoothing=label_smoothing, positive_weight=positive_weight
        )
    return lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)


def loss(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5) -> Dict[str, Any]:
    """Creates a keras-compatible dictionary of loss functions to calculate
    the loss for the contour, note and onset posteriorgrams.

    Args:
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        weighted: Whether or not to use a weighted cross entropy loss.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A dictionary with keys "contour," "note," and "onset" with functions as values to be used to calculate
        transcription losses.

    """
    loss_fn = lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)
    loss_onset = onset_loss(weighted, label_smoothing, positive_weight)
    return {
        "contour": loss_fn,
        "note": loss_fn,
        "onset": loss_onset,
    }

def _initializer() -> jax.nn.initializers.Initializer:
    return jxln.initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")

# TODO: unclear if there is an obvious replication in JAX
#def _kernel_constraint() -> tf.keras.constraints.UnitNorm:
#    return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])


def get_cqt(inputs: jnp.ndarray, n_harmonics: int, use_batchnorm: bool) -> jnp.ndarray:
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = jnp.min(
        jnp.asarray([
            int(jnp.ceil(12.0 * jnp.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ])
    )
    x = nn.FlattenAudioCh()(inputs)
    model = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )
    model.init()
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = signal.NormalizedLog()(x)
    x = jnp.expand_dims(x, -1)
    if use_batchnorm:
        bn = jxln.BatchNorm() # TODO: should I normalize a different way?
        x = bn(x)
        # x = tfkl.BatchNormalization()(x) # TODO: make sure the initiailzation parameters are all the same
    return x


def model(
    n_harmonics: int = 8,
    n_filters_contour: int = 32,
    n_filters_onsets: int = 32,
    n_filters_notes: int = 32,
    no_contours: bool = False,
) -> jxln.Module:
    """Basic Pitch's model implementation.

    Args:
        n_harmonics: The number of harmonics to use in the harmonic stacking layer.
        n_filters_contour: Number of filters for the contour convolutional layer.
        n_filters_onsets: Number of filters for the onsets convolutional layer.
        n_filters_notes: Number of filters for the notes convolutional layer.
        no_contours: Whether or not to include contours in the output.
    """
    # input representation
    # inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))  # (batch, time, ch)
    inputs = jnp.zeros((3, AUDIO_N_SAMPLES, 1))
    x = get_cqt(inputs, n_harmonics, True)

    if n_harmonics > 1:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [0.5] + list(range(1, n_harmonics)),
            N_FREQ_BINS_CONTOURS,
        )(x)
    else:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [1],
            N_FREQ_BINS_CONTOURS,
        )(x)

    # contour layers - fully convolutional
    x_contours = jxln.Conv(
        n_filters_contour,
        (5, 5),
        padding="SAME",
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x)

    x_contours = jxln.BatchNormalization()(x_contours)
    x_contours = jxln.relu(x_contours)

    x_contours = jxln.Conv(
        8,
        (3, 3 * 13),
        padding="SAME",
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x)

    x_contours = jxln.BatchNorm()(x_contours)
    x_contours = jxln.relu(x_contours)

    if not no_contours:
        contour_name = "contour"
        x_contours = jxln.Conv(
            1,
            (5, 5),
            padding="SAME",
            kernel_initializer=_initializer(),
            # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
            name="contours-reduced",
        )(x_contours)
        x_contours = jxln.sigmoid(x_contours)
        x_contours = nn.FlattenFreqCh(name=contour_name)(x_contours)  # contour output

        # reduced contour output as input to notes
        x_contours_reduced = jnp.expand_dims(x_contours, -1)
    else:
        x_contours_reduced = x_contours

    x_contours_reduced = jxln.Conv(
        n_filters_notes,
        (7, 7),
        padding="SAME",
        strides=(1, 3),
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x_contours_reduced)
    x_contours_reduced = jxln.relu(x_contours_reduced)

    # note output layer
    note_name = "note"
    x_notes_pre = jxln.Conv(
        1,
        (7, 3),
        padding="SAME",
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x_contours_reduced)
    x_notes_pre = jxln.sigmoid(x_notes_pre)
    x_notes = nn.FlattenFreqCh(name=note_name)(x_notes_pre)

    # onset output layer

    # onsets - fully convolutional
    x_onset = jxln.Conv(
        n_filters_onsets,
        (5, 5),
        padding="SAME",
        strides=(1, 3),
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x)
    x_onset = jxln.BatchNorm()(x_onset)
    x_onset = jxln.relu(x_onset)
    x_onset = jnp.concatenate([x_notes_pre, x_onset])
    x_onset = jxln.Conv(
        1,
        (3, 3),
        padding="SAME",
        kernel_initializer=_initializer(),
        # kernel_constraint=_kernel_constraint(),  #TODO: fix here once updated
    )(x_onset)
    x_onset = jxln.sigmoid(x_onset)

    onset_name = "onset"
    x_onset = nn.FlattenFreqCh(
        name=onset_name,
    )(x_onset)

    outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}

    return jxln.Module(inputs=inputs, outputs=outputs) # TODO: is this correct? 
