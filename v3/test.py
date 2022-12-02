from jax import value_and_grad
import jax
import jax.numpy as jnp
import haiku as hk
from loss import loss
from cqt_and_hs import harmonic_stacking, load_and_cqt

def main():
    audio_path = "test.m4a"
    audio_tensor = load_and_cqt(audio_path)

    def new_f(audio_tensor, is_training):
        bn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, name="bn")
        normed = bn(audio_tensor, is_training)
        hs = harmonic_stacking(normed)
        return hs
    model = hk.transform_with_state(new_f)
    rng = jax.random.PRNGKey(0)
    params, state = model.init(rng, audio_tensor, True)
    out, state = model.apply(params, state, rng=rng, audio_tensor=audio_tensor, is_training=True)

    epochs = 1000
    learning_rate = jnp.array(0.001)

    def UpdateWeights(weights,gradients):
        return weights - learning_rate * gradients

    def loss_wrapper(params, x, y):
        nonlocal state
        out, state = model.apply(params, state, rng=rng, audio_tensor=x, is_training=True)
        loss_fns = loss()
        return loss_fns["contour"](y[0], out[0]) + loss_fns["note"](y[1], out[1]) + loss_fns["onset"](y[2], out[2])


    for i in range(1, epochs+1):
        loss, param_grads = value_and_grad(loss_wrapper)(params, out, out)
        params = jax.tree_map(UpdateWeights, params, param_grads)

        if i%100 == 0:
            print("MSE : {:.2f}".format(loss))

if __name__ == "__main__":
    main()