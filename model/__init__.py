import numpy as np
import tensorflow as tf

from .wavenet import WaveNet

class DiffWave(tf.keras.Model):
    """DiffWave: A Versatile Diffusion Model for Audio Synthesis.
    Zhifeng Kong et al., 2020.
    *** Slighly modified version of the original DiffWave code to a
    ccount for the singing voice separation approach. If re-using this
    code make sure to reference DiffWave!
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(DiffWave, self).__init__()
        self.config = config
        self.wavenet = WaveNet(config)

    def call(self, signal):
        """Generate denoised audio.
        Args:
            signal: tf.Tensor, [B, T], starting signal for transformation.
        Returns:
            signal: tf.Tensor, [B, T], predicted output.
        """
        alpha = 1 - self.config.beta()
        alpha_bar = np.cumprod(alpha)
        base = tf.ones([tf.shape(signal)[0]], dtype=tf.int32)
        for t in range(self.config.iter, 0, -1):
            eps = self.pred_noise(signal, base * t)
            mu = self.pred_signal(signal, eps, alpha[t - 1], alpha_bar[t - 1])
            signal = mu
        return signal

    def diffusion(self, perturbation, target, pert_to_est, alpha_bar):
        if isinstance(alpha_bar, tf.Tensor):
            alpha_bar = alpha_bar[:, None]
        return tf.sqrt(alpha_bar) * target + \
            tf.sqrt(1 - alpha_bar) * perturbation, pert_to_est

    def pred_noise(self, signal, timestep):
        """Predict noise from signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], timesteps of current markov chain.
        Returns:
            tf.Tensor, [B, T], predicted noise.
        """
        return self.wavenet(signal, timestep)

    def pred_signal(self, signal, eps, alpha, alpha_bar):
        """Compute mean of denoised signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            eps: tf.Tensor, [B, T], estimated noise.
            alpha: float, 1 - beta.
            alpha_bar: float, cumprod(1 - beta).
        Returns:
            tuple,
                mean: tf.Tensor, [B, T], estimated mean of denoised signal.
        """
        signal = tf.dtypes.cast(signal, tf.float64)
        eps = tf.dtypes.cast(eps, tf.float64)

        # Compute mean (our estimation) from the original diffusion parametrization
        mean = (signal - (1 - alpha) / tf.dtypes.cast(tf.sqrt(1 - alpha_bar), tf.float64) * eps) \
            / tf.dtypes.cast(tf.sqrt(alpha), tf.float64)
        return mean

    def write(self, path, optim=None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to write.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path, optim=None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to restore.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)
