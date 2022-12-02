import jax.numpy as jnp
import librosa

harmonics = jnp.array([0.5] + list(range(1, 8)))
def harmonic_stacking(x, bins_per_semitone=1, harmonics=harmonics, n_output_freqs=1):
    """Downsample frequency by stride, upsample channels by 4."""
    shifts = [int(jnp.round(12.0 * bins_per_semitone * jnp.log2(float(h)))) for h in harmonics]
    channels = []
    for shift in shifts:
        if shift == 0:
            padded = x
        elif shift > 0:
            paddings = jnp.array([[0, 0], [0, 0], [0, shift], [0, 0]])
            padded = jnp.pad(x[:, :, shift:, :], paddings)
        elif shift < 0:
            paddings = jnp.array([[0, 0], [0, 0], [-shift, 0], [0, 0]])
            padded = jnp.pad(x[:, :, :shift, :], paddings)
        else:
            raise ValueError

        channels.append(padded)
    x = jnp.concatenate(channels, axis=-1)
    x = x[:, :, : n_output_freqs, :]  # return only the first n_output_freqs frequency channels
    return x

def load_and_cqt(audio_path, sr=22050, hop_length=512, bins_per_octave=12, n_bins=84):
    audio, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(audio.reshape(1, audio.shape[0]), sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
    return jnp.expand_dims(jnp.array(cqt), axis=-1)
