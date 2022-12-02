import jax.numpy as jnp
import optax
from typing import Callable, Dict, Any

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


def loss_dict(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5) -> Dict[str, Any]:
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