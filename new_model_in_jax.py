import jax 
import jax.numpy as jnp


# basic outline:
# step 1: constant q stacking
# step 2: harmonic stacking
 

# def onset_posteriogram_model:
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
    
