from jax import value_and_grad
import jax
import jax.numpy as jnp
import haiku as hk
from loss import loss_dict
import optax
from cqt_and_hs import harmonic_stacking, load_and_cqt
from new_model_in_jax import PosteriorgramModel

def main():
    audio_path = "drive/MyDrive/badpitches/v3/test.m4a"
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

    # print(jax.random.normal(rng, audio_tensor.shape))
    noisy_audio = audio_tensor + jax.random.normal(rng, audio_tensor.shape)

    epochs = 1000
    learning_rate = 0.01
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)


    def update_weights(weights,gradients):
        return optimizer.update(gradients, weights)

    def loss_wrapper(params, state, x, y):
        out, new_state = model.apply(params, state, rng=rng, audio_tensor=x, is_training=True)
        loss_fns = loss_dict()
        loss_yp = jnp.sum(loss_fns["contour"](y[0], out[0]))
        loss_yn = jnp.sum(loss_fns["note"](y[1], out[1]))
        loss_yo = jnp.sum(loss_fns["onset"](y[2], out[2]))
        loss = loss_yp + loss_yn + loss_yo
        return loss, (loss, new_state)



    def step(params, opt_state, state, x, y):
        # loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
        grads, (loss, state) = jax.grad(loss_wrapper, has_aux=True)(params, state, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, state, loss
    
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)  
    
    for i in range(1, epochs+1):
        # grads, (loss, state) = jax.grad(loss_wrapper, has_aux=True)(params, state, noisy_audio, out)
        # params = jax.tree_map(update_weights, params, grads)

        params, opt_state, state, loss = step(params, opt_state, state, noisy_audio, out)
        if i % 100 == 0:
          print("MSE : {:.9f}".format(loss))

if __name__ == "__main__":
    main()
