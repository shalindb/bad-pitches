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

class PosteriorgramModel(hk.Module):

    def __init__(self):
        # define blocks here:
        super().__init__()
        # self.cqt_harmonic_stacking = hk.Sequential([
        #     # cqt layer
        #     # harmonic stacking layer
        #     hk.IdentityCore()  # TODO: replace with actual signal processing
        # ])
        self.cqt_harmonic_stacking = hk.Linear(1)
        
        self.top_branch = hk.Sequential([
            # 32-depth Conv2D (5x5), stride 1x3
            # batch norm
            # ReLU
            hk.Conv2D(32, (5, 5), stride=(1, 3)),
            # hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
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
            # hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
            jax.nn.relu,
            hk.Conv2D(8, (3, 39)),
            # hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True),
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

    def __call__(self, audio) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # preprocessed = self.cqt_harmonic_stacking(audio)
        yp = self.yp_branch(audio)
        yn = self.yn_branch(yp)
        top = self.top_branch(audio)
        concat = jax.numpy.concatenate([top, yn], axis=2)  # TODO: fix this probably
        yo = self.yo_branch(concat)
        print(yp, yn, yo)
        return yp, yn, yo

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
