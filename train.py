import os
import mir_eval
import argparse
import json
import math
import tqdm

import numpy as np
import tensorflow as tf

#from augmentation_utils import time_augment, pitch_augment

from config import Config
from dataset import MUSDB
from model import DiffWave

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer:
    """WaveGrad trainer.
    """
    def __init__(self, model, dataset, config):
        """Initializer.
        Args:
            model: DiffWave, diffwave model.
            dataset: Dataset, input dataset to train the diffusion model
                which provides already batched and normalized speech dataset.
            config: Config, unified configurations.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        self.split = config.train.split // config.data.batch
        self.trainset = self.dataset.dataset().take(self.split) \
            .shuffle(config.train.bufsiz) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.testset = self.dataset.test_dataset() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.optim = tf.keras.optimizers.Adam(
            config.train.lr(),
            config.train.beta1,
            config.train.beta2,
            config.train.eps)

        self.ckpt_intval = config.train.ckpt_intval // config.data.batch

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, "train"))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, "test"))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.alpha = 1 - config.model.beta()
        self.alpha_bar = np.cumprod(self.alpha)

    def compute_loss(self, mixture, vocals, accomp, target="vocals"):
        """Compute loss for noise estimation.
        Args:
            mixture: tf.Tensor, [B, T], raw audio signal mixture.
            vocals: tf.Tensor, [B, T], raw audio signal vocals.
            accomp: tf.Tensor, [B, T], raw audio signal accompaniment.
            target: str, indicating for which target the model is trained.
        Returns:
            loss: tf.Tensor, [], L1-loss between noise and estimation.
        """
        bsize = tf.shape(vocals)[0]
        # [B]
        timesteps = tf.random.uniform(
            [bsize], 1, self.config.model.iter + 1, dtype=tf.int32)
        # [B]
        noise_level = tf.gather(self.alpha_bar, timesteps - 1)
        # [B, T], [B, T]
        if target == "vocals":
            noised, noise = self.model.diffusion(mixture, vocals, accomp, noise_level)
        else:
            noised, noise = self.model.diffusion(mixture, accomp, vocals, noise_level)
        # [B, T]
        eps = self.model.pred_noise(noised, timesteps)
        # []
        loss = tf.reduce_mean(tf.abs(eps - noise))
        return loss

    def train(self, step=0):
        """Train wavegrad.
        Args:
            step: int, starting step.
            ir_unit: int, log ir units.
        """
        best_SDR = 0
        best_step = 0

        # Start training
        print("\n \n")  ## Just to separate bars from Tensorflow-related warnings
        pbar_gen = tqdm.trange(step // self.split, self.config.train.epoch)
        for _ in pbar_gen:
            pbar_gen.set_description("General training process")
            train_loss = []
            with tqdm.tqdm(total=self.split, leave=False) as pbar:
                pbar.set_description("Training epoch ({} steps)".format(self.split))
                for mixture, vocal, accomp in self.trainset:

                    # APPLY DATA AUGMENTATION HERE IF DESIRED
                    #mixture, vocal, accomp = self.pitch_augment(mixture, vocal, accomp)
                    #mixture, vocal, accomp = self.time_augment(mixture, vocal, accomp)

                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_variables)
                        loss = self.compute_loss(
                            mixture,
                            vocal,
                            accomp,
                            target=self.config.train.target)
                        train_loss.append(loss)

                    grad = tape.gradient(loss, self.model.trainable_variables)
                    self.optim.apply_gradients(
                        zip(grad, self.model.trainable_variables))

                    norm = tf.reduce_mean([tf.norm(g) for g in grad])
                    del grad

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {"loss": loss.numpy().item(),
                         "step": step,
                         "grad": norm.numpy().item()})

                    if step % self.ckpt_intval == 0:
                        self.model.write(
                            "{}_{}.ckpt".format(self.ckpt_path, step),
                            self.optim)

            train_loss = sum(train_loss) / len(train_loss)
            validation_loss = []
            for mixture, vocal, accomp in self.testset:
                actual_loss = self.compute_loss(
                    mixture,
                    vocal,
                    accomp,
                    target=self.config.train.target).numpy().item()
                validation_loss.append(actual_loss)

            del vocal, accomp
            validation_loss = sum(validation_loss) / len(validation_loss)

            with self.test_log.as_default():
                if step > 150000:
                    best_SDR, best_step = self.eval(
                        best_SDR,
                        best_step,
                        step,
                        target=self.config.train.target)

            print("==> Current train loss: {}, and validation loss: {}".format(train_loss, validation_loss))
            del train_loss, validation_loss


    def eval(self, best_SDR, best_step, step, target="vocals"):
        """Generate evaluation purpose audio.
        Returns:
            speech: np.ndarray, [T], ground truth.
            pred: np.ndarray, [T], predicted.
            ir: List[np.ndarray], config.model.iter x [T],
                intermediate representations.
        """
        # [T]
        sdr_target = []
        pbar_val = tqdm.tqdm(self.dataset.validation())
        for mixture, vocals, accomp in pbar_val:
            pbar_val.set_description("Validating model")
            # Prepare data for eval
            hop = self.config.data.hop
            nearest_hop = hop * math.floor(mixture.shape[1]/hop)
            mixture_analyze = mixture[:, :nearest_hop]
            vocals_analyze = vocals[:, :nearest_hop]
            accomp_analyze = accomp[:, :nearest_hop]

            if target == "vocals":
                gt_target = vocals_analyze
                gt_rest = accomp_analyze
            else:
                gt_target = accomp_analyze
                gt_rest = vocals_analyze

            # Check vocal track is not silent
            if tf.reduce_max(gt_target).numpy() != 0.0:
                gt_target = tf.squeeze(gt_target, axis=0).numpy()
                gt_rest = tf.squeeze(gt_rest, axis=0).numpy()

                # Predict
                pred_target = self.model(mixture_analyze)

                # Get accompaniment by substraction
                mixture_analyze = tf.squeeze(mixture_analyze, axis=0).numpy()
                pred_target = tf.squeeze(pred_target, axis=0).numpy()
                pred_target = pred_target * (np.max(np.abs(gt_target)) / np.max(np.abs(pred_target)))
                pred_rest = mixture_analyze - pred_target

                # Evaluate
                ref = np.array([gt_target, gt_rest])
                est = np.array([pred_target, pred_rest])
                sdr, _, _, _, _ = mir_eval.separation.bss_eval_images(
                    ref, est, compute_permutation=False)
                sdr_target.append(sdr[0])

        # Updating best new model taking SDR into account
        if np.median(sdr_target) > best_SDR:
            print("Saving best new model with SDR: {}".format(str(np.median(sdr_target))))
            self.model.write("{}_BEST-MODEL_{}.ckpt".format(self.ckpt_path, str(step)), self.optim)
            best_SDR = np.median(sdr_target)
            best_step = step
        else:
            print("Current best model: {} from step {}".format(str(best_SDR), str(best_step)))
            print("The median SDR of this evaluation step is: {}".format(np.median(sdr_target)))
        return best_SDR, best_step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--load-step", default=0, type=int)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config = Config()
    if args.config is not None:
        print("[*] load config: " + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    log_path = os.path.join(config.train.log, config.train.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    dataset = MUSDB(config.data, data_dir=args.data_dir)
    diffwave = DiffWave(config.model)
    trainer = Trainer(diffwave, dataset, config)

    if args.load_step > 0:
        super_path = os.path.join(config.train.ckpt, config.train.name)
        ckpt_path = "{}_{}.ckpt".format(config.train.name, args.load_step)
        ckpt_path = next(
            name for name in os.listdir(super_path)
                 if name.startswith(ckpt_path) and name.endswith(".index"))
        ckpt_path = os.path.join(super_path, ckpt_path[:-6])
        
        print("[*] load checkpoint: " + ckpt_path)
        trainer.model.restore(ckpt_path, trainer.optim)

    with open(os.path.join(config.train.ckpt, config.train.name + ".json"), "w") as f:
        json.dump(config.dump(), f)

    trainer.train(args.load_step)
