from flax import linen as nn
import jax.numpy as jnp

class LambdaLayer(nn.Module):
    def setup(self, fn):
        self.fn = fn
    
    def __call__(self, x):
        return self.fn(x)
    

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

