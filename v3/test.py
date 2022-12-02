from jax import value_and_grad
import jax
import jax.numpy as jnp
import haiku as hk
from loss import loss_dict
from cqt_and_hs import harmonic_stacking, load_and_cqt
from new_model_in_jax import PosteriorgramModel

def main():
    audio_path = "test.m4a"
    audio_tensor = load_and_cqt(audio_path)

    def new_f(audio_tensor, is_training):
        bn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9, name="bn")
        normed = bn(audio_tensor, is_training)
        hs = harmonic_stacking(normed)
        m = PosteriorgramModel()
        out = m(hs, is_training)
        return out
    model = hk.transform_with_state(new_f)
    rng = jax.random.PRNGKey(42)
    params, state = model.init(rng, audio_tensor, True)
    out, state = model.apply(params, state, rng=rng, audio_tensor=audio_tensor, is_training=True)

    noisy_audio = audio_tensor + 1000 #jax.random.normal(rng, audio_tensor.shape) * 1000

    epochs = 1000
    learning_rate = jnp.array(0.001)

    def UpdateWeights(weights,gradients):
        return weights - learning_rate * gradients

    def loss_wrapper(params, state, x, y):
        out, new_state = model.apply(params, state, rng=rng, audio_tensor=x, is_training=True)
        loss_fns = loss_dict()
        loss_yp = jnp.sum(loss_fns["contour"](y[0], out[0]))
        loss_yn = jnp.sum(loss_fns["note"](y[1], out[1]))
        loss_yo = jnp.sum(loss_fns["onset"](y[2], out[2]))
        loss = loss_yp + loss_yn + loss_yo
        return loss, (loss, new_state)


    for i in range(1, epochs+1):
        grads, (loss, state) = jax.grad(loss_wrapper, has_aux=True)(params, state, noisy_audio, out)
        params = jax.tree_map(UpdateWeights, params, grads)

        print("MSE : {:.9f}".format(loss))

if __name__ == "__main__":
    main()