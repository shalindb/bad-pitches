from typing import Union, List, Dict, Tuple, Optional
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import librosa
import pathlib


# basic outline:
# step 1: constant q stacking # Eran?
# step 2: harmonic stacking # Eran?

class CQTHarmonicStacking(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # step 1: constant q stacking (or is it preprocessed?)
        # step 2: harmonic stacking
        return x
    
class TopBranch(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv32_5_5 = hk.Conv2D(32, (5, 5), stride=(1, 3), name="conv32_5_5")
        self.bn = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True, name="bn")
        
    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x = self.conv32_5_5(x)
        x = self.bn(x, is_training=is_training)
        x = jax.nn.relu(x)
        return x

    def load_weights(self, var_dict: dict):
        """
        vd["conv2d_4"] = conv32_5_5(x)
        vd["batch_normalization_3"] = bn
        vd["conv2d_5"] = ???
        """
        pass

class YpBranch(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv16_5_5 = hk.Conv2D(16, (5, 5), name="conv16_5_5")
        # self.bn1 = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True, name="bn1")
        self.conv8_3_39 = hk.Conv2D(8, (3, 39), name="conv8_3_39")
        # self.bn2 = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True, name="bn2")
        self.conv_1_5_5 = hk.Conv2D(1, (5, 5), name="conv_1_5_5")
        
    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        x = self.conv16_5_5(x)
        # x = self.bn1(x, is_training=is_training)
        x = jax.nn.relu(x)
        x = self.conv8_3_39(x)
        # x = self.bn2(x, is_training=is_training)
        x = jax.nn.relu(x)
        x = self.conv_1_5_5(x)
        x = jax.nn.sigmoid(x)
        return x

    def load_weights(self, var_dict: dict):
        """
        vd["conv2d_1"] = conv8_3_39
        vd["batch_normalization_2"] = bn2
        vd["contours-reduced"]] = conv_1_5_5
        """
        pass

class YnBranch(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv32_7_7 = hk.Conv2D(32, (7, 7), stride=(1, 3), name="conv32_7_7")
        self.conv1_7_3 = hk.Conv2D(1, (7, 3), name="conv1_7_3")
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv32_7_7(x)
        x = jax.nn.relu(x)
        x = self.conv1_7_3(x)
        x = jax.nn.sigmoid(x)
        return x
    def load_weights(self, var_dict: dict):
        """
        vd["conv2d_2"] =  conv32_7_7
        vd["conv2d_3"] = conv1_7_3
        """
        pass

class YoBranch(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.conv1_3_3 = hk.Conv2D(1, (3, 3), name="conv1_3_3")
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv1_3_3(x)
        x = jax.nn.sigmoid(x)
        return x

    def load_weights(self, var_dict: dict):
        """
        vd["conv2d_5"] = conv1_3_3
        """
        pass
    
class PosteriorgramModel(hk.Module):

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.cqt_harmonic_stacking = CQTHarmonicStacking(name="cqt_harmonic_stacking")
        self.top_branch = TopBranch(name="top_branch")
        self.yp_branch = YpBranch(name="yp_branch")
        self.yn_branch = YnBranch(name="yn_branch")
        self.yo_branch = YoBranch(name="yo_branch")

    def __call__(self, audio: jnp.ndarray, is_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        processed = self.cqt_harmonic_stacking(audio)
        yp = self.yp_branch(processed, is_training)
        yn = self.yn_branch(yp)
        top = self.top_branch(processed, is_training)
        concat = jax.numpy.concatenate([top, yn], axis=-1) 
        yo = self.yo_branch(concat)
        return yp, yn, yo

    
    def load_weights(self, model_dict):
        variable_dict = {}
        for elem in model_dict: # store all elements by name
            var_name = elem._handle_name.split("/")[0]
            if var_name in variable_dict:
                variable_dict[var_name].append(elem)
            else:
                variable_dict[var_name] = [elem]
        
        # pass along variables to each branch/block:
        # cqt_vars = {}
        yp_vars = {}
        yn_vars = {}
        top_vars = {}

        for var_name, elems in variable_dict.items():
            # if var_name in ["batch_normalization"]: # cqt
            #    cqt_vars[var_name] = elems
            if var_name in ["conv2d_1", "batch_normalization_2", "contours-reduced"]: # yp
                yp_vars[var_name] = elems
            if var_name in ["conv2d_2", "conv2d_3"]: # yn
                yn_vars[var_name] = elems
            if var_name in ["conv2d_4", "batch_normalization_3", "conv2d_5"]: # top_vars
                top_vars[var_name] = elems

        # self.cqt_harmonic_stacking.load_weights(cqt_vars)
        self.yp_branch.load_weights(yp_vars)
        self.yn_branch.load_weights(yn_vars)
        self.top_branch.load_weights(top_vars)
        self.yo_branch.load_weights(top_vars)


