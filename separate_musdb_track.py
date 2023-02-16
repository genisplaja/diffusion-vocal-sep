"""
This file performs separation of a musdb file, assuming the structure of the dataset is correct. The user inputs the mixture that is going to
be separated, and the corresponding singing voice and accompaniment are used to provide, in addition to the separated sources, the SDR metric
that the model has achieved on the selected track.
"""

import os
import math
import glob
import tqdm
import json
import norbert
import librosa
import argparse

import numpy as np
import soundfile as sf
import tensorflow as tf

from config import Config
from model import DiffWave

SAMPLING_RATE = 22050


def get_window(signal, boundary=None):
    window_out = np.ones(signal.shape)
    midpoint = window_out.shape[0] // 2
    if boundary == "start":
        window_out[midpoint:] = np.linspace(1, 0, window_out.shape[0]-midpoint)
    elif boundary == "end":
        window_out[:midpoint] = np.linspace(0, 1, window_out.shape[0]-midpoint)
    else:
        window_out[:midpoint] = np.linspace(0, 1, window_out.shape[0]-midpoint)
        window_out[midpoint:] = np.linspace(1, 0, window_out.shape[0]-midpoint)
    return window_out

def my_special_round(x, base):
    return math.ceil(base * round(float(x)/base))

def GlobalSDR(references, separations):
    """ Global SDR: main (or standard) metric from SiSEC 2021 and MDX"""
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - separations), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(os.path.join(".", "ckpt", args.model_name + ".json")):
        return ValueError("Please make sure the model exists and have a config file in ./ckpt")
    if not args.input_file:
        return ValueError("Please enter the input file through the --input-file argument")
    if not os.path.exists(args.input_file):
        return ValueError("Input file not found: please make sure the input file exists!")

    # prepare directory for samples
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(".", "ckpt", args.model_name + ".json"), "r") as f:
        config = Config.load(json.load(f))

    # Create and initialize the model
    config = Config()
    diffwave = DiffWave(config.model)
    if args.ckpt is not None:
        diffwave.restore(args.ckpt).expect_partial()
    else:
        ckpts = glob.glob(os.path.join(".", "ckpt", args.model_name + "*.ckpt-1.data-00000-of-00001"))
        ckpts = [x for x in ckpts if "BEST-MODEL" in x]
        latest_step = np.max([float(x.split("_")[-1].replace(".ckpt-1.data-00000-of-00001")) for x in ckpts])
        ckpt_path = os.path.join(".", "ckpt", args.model_name, \
            args.model_name + "_BEST-MODEL_" + str(latest_step) + ".ckpt-1.data-00000-of-00001")
        diffwave.restore(ckpt_path).expect_partial()

    print("Separating: {}".format(args.input_file))
    filename = args.input_file.split("/")[-1]
    mixture = tf.io.read_file(args.input_file)
    mixture, sr =  tf.audio.decode_wav(mixture, desired_channels=1)
    vocals = tf.io.read_file(args.input_file.replace("mixture.wav", "vocals.wav"))

    # Check if accompaniment is available, otherwise we create it
    if os.path.exists(args.input_file.replace("mixture.wav", "accompaniment.wav")):
        accomp = args.input_file.replace("mixture.wav", "accompaniment.wav")
        accomp, sr = tf.audio.decode_wav(accomp, desired_channels=1)
    else:
        bass = tf.audio.decode_wav(tf.io.read_file(args.input_file.replace("mixture.wav", "bass.wav")))
        drums = tf.audio.decode_wav(tf.io.read_file(args.input_file.replace("mixture.wav", "drums.wav")))
        other = tf.audio.decode_wav(tf.io.read_file(args.input_file.replace("mixture.wav", "other.wav")))
        accomp = bass + drums + other

    vocals, sr =  tf.audio.decode_wav(vocals, desired_channels=1)

    if sr != SAMPLING_RATE:
        return ValueError("Please resample MUSDB audio to {}Hz before running inference.".format(SAMPLING_RATE))

    mixture = tf.squeeze(mixture, axis=-1)
    mixture_hopped_shape = math.ceil(mixture.shape[0] / config.data.hop) * config.data.hop
    output_vocals = np.zeros(mixture_hopped_shape)
    output_accomp = np.zeros(mixture_hopped_shape)
    hopsized_batch = ((int(args.batch)*SAMPLING_RATE) / 2) // config.data.hop * config.data.hop
    sec = math.floor(mixture_hopped_shape / hopsized_batch)

    for trim in tqdm.tqdm(np.arange(sec)):
        trim_low = int(trim*hopsized_batch)
        trim_high = int(trim_low + (hopsized_batch*2))
        mixture_analyse = mixture[trim_low:trim_high]
        vocals_analyse = vocals[trim_low:trim_high]

        # Last batch (might be shorter than hopsized batch sized)
        if mixture_analyse.shape[0] < hopsized_batch*2:
            padded_len = my_special_round(mixture_analyse.shape[0], base=config.data.hop)
            difference = int(padded_len - mixture_analyse.shape[0])
            mixture_analyse = tf.concat([mixture_analyse, tf.zeros([difference])], axis=0)

        output_signal = diffwave(mixture_analyse[None])
        pred_audio = tf.squeeze(output_signal, axis=0).numpy()

        mixture_analyse = mixture_analyse.numpy()
        pred_audio = pred_audio * (np.max(np.abs(vocals_analyse)) / np.max(np.abs(pred_audio)))
        pred_accomp = mixture_analyse - pred_audio

        if args.wiener:
            pred_audio = np.squeeze(pred_audio, axis=0)

            # Compute stft
            vocal_spec = np.transpose(librosa.stft(pred_audio), [1, 0])
            accomp_spec = np.transpose(librosa.stft(pred_accomp), [1, 0])

            # Separate mags and phases
            vocal_mag = np.abs(vocal_spec)
            vocal_phase = np.angle(vocal_spec)
            accomp_mag = np.abs(accomp_spec)
            accomp_phase = np.angle(accomp_spec)

            # Preparing inputs for wiener filtering
            mix_spec = np.transpose(librosa.stft(mixture_analyse), [1, 0])
            sources = np.transpose(np.vstack([vocal_mag[None], accomp_mag[None]]), [1, 2, 0])
            mix_spec = np.expand_dims(mix_spec, axis=-1)
            sources = np.expand_dims(sources, axis=2)

            # Wiener
            specs = norbert.wiener(sources, mix_spec)

            # Building output specs with filtered mags and original phases
            vocal_spec = np.abs(np.squeeze(specs[:, :, :, 0], axis=-1)) * np.exp(1j * vocal_phase)
            accomp_spec = np.abs(np.squeeze(specs[:, :, :, 1], axis=-1)) * np.exp(1j * accomp_phase)
            pred_audio = librosa.istft(np.transpose(vocal_spec, [1, 0]))
            pred_accomp = librosa.istft(np.transpose(accomp_spec, [1, 0]))
            pred_audio = np.squeeze(pred_audio, axis=0)
            pred_accomp = np.squeeze(pred_accomp, axis=0)

        # Get boundary 
        boundary = None
        boundary = "start" if trim == 0 else None
        boundary = "end" if trim == sec-1 else None

        placehold_voc = np.zeros(output_vocals.shape)
        placehold_acc = np.zeros(output_accomp.shape)
        placehold_voc[trim_low:trim_high] = pred_audio * get_window(pred_audio, boundary=boundary)
        placehold_acc[trim_low:trim_high] = pred_accomp * get_window(pred_accomp, boundary=boundary)
        output_vocals += placehold_voc
        output_accomp += placehold_acc

    output_vocals = output_vocals[:mixture.shape[0]]
    output_accomp = output_accomp[:mixture.shape[0]]

    scores = GlobalSDR(np.array([vocals, accomp]), np.array([output_vocals, output_accomp])[..., None])
    print("VOCALS ==> SDR:", scores[0])
    print("ACCOMP ==> SDR:", scores[1])

    # Write output to file
    sf.write(os.path.join(args.output_dir, filename.replace(".wav", "_separated-vocals.wav"), output_vocals, SAMPLING_RATE))
    sf.write(os.path.join(args.output_dir, filename.replace(".wav", "_separated-accompaniment.wav"), output_accomp, SAMPLING_RATE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-dir", default="./output/")
    parser.add_argument("--model-name", default="20-step-vocal")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch", default=20)
    parser.add_argument("--wiener", default=False)
    parser.add_argument("--gpu", default=-1)
    args = parser.parse_args()
    main(args)
