# diffusion-vocal-sep
Code for training and inferencing using the method presented in "A diffusion-inspired training strategy for singing voice extraction in the waveform domain", presented in ISMIR 2022.

This code is an adaptation of [DiffWave code](https://github.com/revsic/tf-diffwave). If re-using the implementation itself, please refer to
the original repository of DiffWave.


## Quick reference
### Data pre-processing

To enhance computation, the model is configured to be trained at ``22050Hz``. You may change that if desired. Bear in mind that at some parts of the code,
this sampling rate is hard-coded. To train the dataset, you first need to pre-process MUSDB18HQ, to resample, create the accompaniments, and trim the 
recordings into chunks of 4 seconds, in order to enhance the training stage. You can do that by running:

```python
python3 process_musdb.py --data-dir </path/to/musdb18hq/> --output-dir </path/to/output/folder>  --train <True or False>
```

If ``--train`` is ``False``, in order to pre-process the testing data, the tracks will be resampled but not chunked, which is better for evaluation.


### Training

To start the training, run:

```python
python3 train.py --data-dir </path/to/train/dataset/>
```

If the training is interrupted, you can continue training by using:

```python
python3 train.py --data-dir </path/to/training/dataset/> --config </path/to/config.json>  --load-step <training-step-to-load> 
```

The ``config.json`` file is stored in ``./ckpt/`` by default. The weights are stored, by default, in 
``./ckpt/<model-name>/<model-name>_<training-step>.ckpt-1.data-00000-of-00001``. If the training process
is stopped and you need to continue from where you left off, by setting the argument ``--load-step``, the training code takes the closest stored
step. Note that you can manually set, in the ``config.py`` file, how often (in terms of steps) the weights are stored. 

We also store the model weights that obtain the best source separation metrics in the validation runs. This model is store in the
following format: ``./ckpt/<model-name>/<model-name>_BEST_MODEL_<training-step>.ckpt-1.data-00000-of-00001``.

Each model you train has a particular name, which is set in the ``config.py`` file, and will be used to relate the configuration, stored weights,
and will be useful when inferencing.

In the main configuration file, you can also set up the target you want to train the model for. Bear in mind that there are configuration files also
for the dataloader and the model structure (which are found, respectively, in ``./dataset/`` and ``./model/``).


### Inference

To run inference on a particular recording, run:

```python
python3 separate.py --input-file </path/to/input/file> --output-dir </folder/where/to/save/output> --model-name <name-of-model> --batch <duration-of-chunks> --wiener <True or False> 
```

The ``separate.py`` function also takes a ``--ckpt`` parameter, which you can set manually. If ``--ckpt`` is not set, the path is built from the
given ``--model-name`` (which is required), and the ``BEST_MODEL`` for the last available step is taken. During inference, the audio array to use
as input is chunked used a pre-defined size of 20 seconds, but the user can select any duration of the chunks. This is done to prevent filling the
available memory.

We include an additional file to run inference on a MUSDB18HQ track. Basically, assuming that the folder structure is that of MUSDB18HQ, you can run
the file as specified below, providing the path to the ``mixture.wav`` in MUSDB you would like to run inference on. The function will automatically 
take the references to compute, for this particular track, the SDR metric using the default implementation that is used in the MDX Challenge 2021-2023.

```python
python3 separate_musdb_track.py --input-file </path/to/input/file> --output-dir </folder/where/to/save/output> --model-name <name-of-model> --batch <duration-of-chunks> --wiener <True or False> 
```


### Citing

```
"A diffusion-inspired training strategy for singing voice extraction in the waveform domain"
Genís Plaja-Roglans, Marius Miron, Xavier Serra
in Proceedings of the International Society for Music Information Retrieval (ISMIR) Conference, 2022 (Bengaluru, India)
```

```
@inproceedings{
  Plaja-Roglans_2022,
  title={A diffusion-inspired training strategy for singing voice extraction in the waveform domain},
  author={Plaja-Roglans, Genís and Miron, Marius and Serra, Xavier},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2022}
}
```

Once again, this implementation is broadly based on the [TensorFlow implementation of DiffWave](https://github.com/revsic/tf-diffwave). If you are
willing to use parts of the implementation itself, we kindly request that you refer to the TensorFlow DiffWave release and also cite it.
