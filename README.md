# Pytorch Conformer
Pytorch implementation of [conformer](https://arxiv.org/abs/2005.08100) model. Includes code to train/validate a speech recognition model on the LibriSpeech dataset.

## Usage

### Train model from scratch:
```
python train.py --data_dir=./data --train_set=train-clean-100 --test_set=test_clean --checkpoint_path=model_best.pt
```
### Resume training from checkpoint
```
python train.py --load_checkpoint --checkpoint_path=model_best.pt
```
### Train with mixed precision: 
```
python train.py --use_amp
```
For a full list of command line arguments see train.py. For valid train_set and test_set values, see torchaudio's [LibriSpeech dataset](https://pytorch.org/audio/stable/datasets.html). The model parameters default to the Conformer (S) configuration. For the Conformer (M) and Conformer (L) models, refer to the table below: 

<img src="https://jwink-public.s3.amazonaws.com/conformer-params.png" width="500"/>

## Other Implementations
- https://github.com/sooftware/conformer
- https://github.com/lucidrains/conformer

## TODO:
- Language Model (LM) implementation
- Multi-GPU support
- Smart batching
- Support for full LibriSpeech960h train set
- Support for other decoders (ie: transformer decoder, etc.)

