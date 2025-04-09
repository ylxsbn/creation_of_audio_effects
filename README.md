# Creation of Audio Effects Using Deep Learning

This repo hosts the code for the "Creating Audio Effects Using Deep Learning" project, conducted as a coursework at HSE University.

The implemented baseline model is largely based on [General-purpose approach](https://arxiv.org/pdf/1905.06148) paper.

This repo also containts the reworked pipeline of the [Audio-MAE](https://github.com/facebookresearch/AudioMAE) model.

## Demo samples

The demo samples for chorus, tremolo and distortion audio effects are available at [Demo samples](https://ylxsbn.github.io/demos.html) website.

### Installation

Clone this repo and install all the required packages:
```bash
git clone https://github.com/ylxsbn/creation_of_audio_effects.git
pip install -r creation_of_audio_effects/requirements.txt
```

If you want to use the Audio-MAE model, you should also follow the installation instructions from the [original repository](https://github.com/facebookresearch/AudioMAE).

### Preparing data

Download the **IDMT-SMT-Audio-Effects** dataset from [here](https://zenodo.org/records/7544032) and place it in the `data/idmt-smt-dataset` folder.

## Baseline model

You can train and run this model on various audio effects that are available in the dataset.

### Pre-training stage

To pretrain a model specify the desired effect in dataset config and run the following script:

```bash
python train.py -cn=pretrain
```

### Training stage

To train a model specify the desired effect in dataset config and run the following script:

```bash
python train.py -cn=baseline
```

If you want to use the pretrained weights you should specify the `pretrain_path` in train.py script.

### Inference
To evaluate the model you should configure the inference config and run the inference:
```bash
python inference.py
```

## Audio-MAE

### Pipeline

Firstly, be sure to use the modified Audio-MAE forward function:

```python
def forward(self, input_spec, **batch):
    mask_ratio = 0.0
    emb_enc, _, ids_restore, _ = self.forward_encoder(input_spec, mask_ratio, mask_2d=self.mask_2d)
    pred, _, _ = self.forward_decoder(emb_enc, ids_restore)
    pred = self.unpatchify(pred)
    return {"predicted_spec": pred}
```

### Pre-training stage

To pretrain the Audio-MAE model specify the desired effect in dataset config and run the following script:

```bash
python train.py -cn=audiomae_pretrain
```

### Training stage

To train a model specify the desired effect in dataset config and run the following script:

```bash
python train.py -cn=audiomae_train
```

If you want to use the pretrained weights you should specify the pretrain_path in train.py script.

### Inference
To evaluate the model you should specify the inference options in config and run the inference script:
```bash
python inference.py
```

## Checkpoints

Checkpoints for chorus, tremolo and distortion audio effects are available at [here](https://drive.google.com/drive/folders/11vcegwz2z4I4xHN677P1_ZX0ST45x0Em?usp=drive_link)

You can download and put them in `saved/model_name` folder.

You can specify the `resume_from` option in config or `pretrain_path` in script to use these weights.

## Credits


This repository utilizes the [pytorch-template](https://github.com/Blinorot/pytorch_project_template/) and adapts some code from 
[pyneuralfx](https://github.com/ytsrt66589/pyneuralfx) and [facebookresearch](https://github.com/facebookresearch/) for the purpose of creating audio effects.

The lightweight model is based on [General-purpose](https://arxiv.org/pdf/1905.06148) architecture that was proposed by Martinez et al.


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)