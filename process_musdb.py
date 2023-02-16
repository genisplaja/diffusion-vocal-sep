"""
To present more sparse and diverse data batches during training we comply with the training style of
DiffWave and split the music recordings in chunks of 4 seconds. 

To get the proper encoding for TensorFlow to properly read the wav files, we use torchaudio which
includes a very versatile way to get the audio files encoded as such.
"""

import os
import math
import torch
import argparse

import numpy as np
import torchaudio as T

from glob import glob
from tqdm import tqdm

def load_resample_downmix(path, new_sr=22050):
    # Loading
    audio, sr = T.load(path)
    # Resampling
    resampling = T.transforms.Resample(sr, new_sr)
    audio = resampling(audio)
    # Processing
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    return audio


def main_train(data_dir, output_dir, sample_len, sample_rate):

    if (data_dir is None) or (output_dir is None):
        raise ValueError("You must enter both directory of MUSDB18HQ and output directory")

    # Creating output dir if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get path list of songs
    musdb_songs = glob(os.path.join(data_dir, "*/"))

    for song_id, i in tqdm(enumerate(musdb_songs)):
        if T.__version__ > "0.7.0":
            audio_mix = load_resample_downmix(os.path.join(i, "mixture.wav"), sample_rate)
            audio_vocals = load_resample_downmix(os.path.join(i, "vocals.wav"), sample_rate)
            audio_bass = load_resample_downmix(os.path.join(i, "bass.wav"), sample_rate)
            audio_drums = load_resample_downmix(os.path.join(i, "drums.wav"), sample_rate)
            audio_other = load_resample_downmix(os.path.join(i, "other.wav"), sample_rate)

            # Get accomp track
            audio_accomp = audio_drums + audio_bass + audio_other

            audio_mix = torch.clamp(audio_mix, -1.0, 1.0)
            audio_vocals = torch.clamp(audio_vocals, -1.0, 1.0)
            audio_accomp = torch.clamp(audio_accomp, -1.0, 1.0)

            for trim in np.arange(math.floor((audio_mix.shape[1])/(sample_rate*sample_len))):
                audio_mix_trim = audio_mix[
                    :, trim*(sample_rate*sample_len):(trim+1)*(sample_rate*sample_len)
                ]
                audio_voc_trim = audio_vocals[
                    :, trim*(sample_rate*sample_len):(trim+1)*(sample_rate*sample_len)
                ]
                audio_accomp_trim = audio_accomp[
                    :, trim*(sample_rate*sample_len):(trim+1)*(sample_rate*sample_len)
                ]

                # Formatting filename
                if torch.max(audio_voc_trim[0]) == torch.tensor(0.0):
                    track_id =  "silence_" + song_id + "_" + str(trim)
                else:
                    track_id = song_id + "_" + str(trim)

                # Saving
                T.save(
                    os.path.join(output_dir, track_id + "_mixture.wav"),
                    audio_mix_trim.cpu(),
                    sample_rate=sample_rate,
                    bits_per_sample=16
                )
                T.save(
                    os.path.join(output_dir, track_id + "_vocals.wav"),
                    audio_voc_trim.cpu(),
                    sample_rate=sample_rate,
                    bits_per_sample=16
                )
                T.save(
                    os.path.join(output_dir, track_id + "_accompaniment.wav"),
                    audio_accomp_trim.cpu(),
                    sample_rate=sample_rate,
                    bits_per_sample=16
                )

        else:
            raise ModuleNotFoundError("Need a version > 0.7.0 for torchaudio!")


def main_validation(data_dir, output_dir, sample_rate):

    if (data_dir is None) or (output_dir is None):
        raise ValueError("You must enter both directory of MUSDB18HQ test and output directory")

    # Creating output dir if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get path list of songs
    musdb_songs = glob(os.path.join(data_dir, "*/"))

    for i in tqdm(musdb_songs):
        if T.__version__ > "0.7.0":
            song_name = i.split("/")[-2]
            audio_mix = load_resample_downmix(os.path.join(i, "mixture.wav"), sample_rate)
            audio_vocals = load_resample_downmix(os.path.join(i, "vocals.wav"), sample_rate)
            audio_bass = load_resample_downmix(os.path.join(i, "bass.wav"), sample_rate)
            audio_drums = load_resample_downmix(os.path.join(i, "drums.wav"), sample_rate)
            audio_other = load_resample_downmix(os.path.join(i, "other.wav"), sample_rate)

            # Get accomp track
            audio_accomp = audio_drums + audio_bass + audio_other

            audio_mix = torch.clamp(audio_mix, -1.0, 1.0)
            audio_vocals = torch.clamp(audio_vocals, -1.0, 1.0)
            audio_accomp = torch.clamp(audio_accomp, -1.0, 1.0)

            # Saving
            T.save(
                os.path.join(output_dir, song_name, "mixture.wav"),
                audio_mix.cpu(),
                sample_rate=sample_rate,
                bits_per_sample=16
            )
            T.save(
                os.path.join(output_dir, song_name, "vocals.wav"),
                audio_vocals.cpu(),
                sample_rate=sample_rate,
                bits_per_sample=16
            )
            T.save(
                os.path.join(output_dir, song_name, "accompaniment.wav"),
                audio_accomp.cpu(),
                sample_rate=sample_rate,
                bits_per_sample=16
            )

        else:
            raise ModuleNotFoundError("Need a version > 0.7.0 for torchaudio!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-len", default=4)
    parser.add_argument("--sample-rate", default=22050)
    args = parser.parse_args()
    if args.train:
        main_train(args.data_dir, args.output_dir, args.sample_len, args.sample_rate)
    else:
        main_validation(args.data_dir, args.output_dir, args.sample_rate)