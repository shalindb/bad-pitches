from flax import linen as nn
import jax.numpy as jnp

class LambdaLayer(nn.Module):
    def setup(self, fn):
        self.fn = fn
    
    def __call__(self, x):
        return self.fn(x)