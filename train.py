import os
import gc
import argparse
import torchaudio
import torch
import torch.nn.functional as F

from torch import nn
from torchmetrics.text.wer import WordErrorRate
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import ConformerEncoder, LSTMDecoder
from utils import *

parser = argparse.ArgumentParser("conformer")
parser.add_argument('--data_dir', type=str, default='./data', help='location to download data')
parser.add_argument('--checkpoint_path', type=str, default='model_best.pt', help='path to store/load checkpoints')
parser.add_argument('--load_checkpoint', action='store_true', default=False, help='resume training from checkpoint')
parser.add_argument('--train_set', type=str, default='train-clean-100', help='train dataset')
parser.add_argument('--test_set', type=str, default='test-clean', help='test dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--warmup_steps', type=float, default=10000, help='Multiply by sqrt(d_model) to get max_lr')
parser.add_argument('--peak_lr_ratio', type=int, default=0.05, help='Number of warmup steps for LR scheduler')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id (optional)')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--report_freq', type=int, default=100, help='training objective report frequency')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision to train')
parser.add_argument('--attention_heads', type=int, default=4, help='number of heads to use for multi-head attention')
parser.add_argument('--d_input', type=int, default=80, help='dimension of the input (num filter banks)')
parser.add_argument('--d_encoder', type=int, default=144, help='dimension of the encoder')
parser.add_argument('--d_decoder', type=int, default=320, help='dimension of the decoder')
parser.add_argument('--encoder_layers', type=int, default=16, help='number of conformer blocks in the encoder')
parser.add_argument('--decoder_layers', type=int, default=1, help='number of decoder layers')
parser.add_argument('--conv_kernel_size', type=int, default=31, help='size of kernel for conformer convolution blocks')
parser.add_argument('--feed_forward_expansion_factor', type=int, default=4, help='expansion factor for conformer feed forward blocks')
parser.add_argument('--feed_forward_residual_factor', type=int, default=.5, help='residual factor for conformer feed forward blocks')
parser.add_argument('--dropout', type=float, default=.1, help='dropout factor for conformer model')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='model weight decay (corresponds to L2 regularization)')
parser.add_argument('--variational_noise_std', type=float, default=.0001, help='std of noise added to model weights for regularization')
parser.add_argument('--num_workers', type=int, default=2, help='num_workers for the dataloader')
parser.add_argument('--smart_batch', type=bool, default=True, help='Use smart batching for faster training')
parser.add_argument('--accumulate_iters', type=int, default=1, help='Number of iterations to accumulate gradients')
args = parser.parse_args()


def main():

  # Load Data
  if not os.path.isdir(args.data_dir):
    os.mkdir(args.data_dir)
  train_data = torchaudio.datasets.LIBRISPEECH(root=args.data_dir, url=args.train_set, download=True)
  test_data = torchaudio.datasets.LIBRISPEECH(args.data_dir, url=args.test_set, download=True)

  
  if args.smart_batch:
    print('Sorting training data for smart batching...')
    sorted_train_inds = [ind for ind, _ in sorted(enumerate(train_data), key=lambda x: x[1][0].shape[1])]
    sorted_test_inds = [ind for ind, _ in sorted(enumerate(test_data), key=lambda x: x[1][0].shape[1])]
    train_loader = DataLoader(dataset=train_data,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    batch_sampler=BatchSampler(sorted_train_inds, batch_size=args.batch_size),
                                    collate_fn=lambda x: preprocess_example(x, 'train'))

    test_loader = DataLoader(dataset=test_data,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                batch_sampler=BatchSampler(sorted_test_inds, batch_size=args.batch_size),
                                collate_fn=lambda x: preprocess_example(x, 'valid'))
  else:
    train_loader = DataLoader(dataset=train_data,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=lambda x: preprocess_example(x, 'train'))

    test_loader = DataLoader(dataset=test_data,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=lambda x: preprocess_example(x, 'valid'))


  # Declare Models  
  
  encoder = ConformerEncoder(
                      d_input=args.d_input,
                      d_model=args.d_encoder,
                      num_layers=args.encoder_layers,
                      conv_kernel_size=args.conv_kernel_size, 
                      dropout=args.dropout,
                      feed_forward_residual_factor=args.feed_forward_residual_factor,
                      feed_forward_expansion_factor=args.feed_forward_expansion_factor,
                      num_heads=args.attention_heads)
  
  decoder = LSTMDecoder(
                  d_encoder=args.d_encoder, 
                  d_decoder=args.d_decoder, 
                  num_layers=args.decoder_layers)
  char_decoder = GreedyCharacterDecoder().eval()
  criterion = nn.CTCLoss(blank=28, zero_infinity=True)
  optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4, betas=(.9, .98), eps=1e-05 if args.use_amp else 1e-09, weight_decay=args.weight_decay)
  scheduler = TransformerLrScheduler(optimizer, args.d_encoder, args.warmup_steps)

  # Print model size
  model_size(encoder, 'Encoder')
  model_size(decoder, 'Decoder')

  gc.collect()

  # GPU Setup
  if torch.cuda.is_available():
    print('Using GPU')
    gpu = True
    torch.cuda.set_device(args.gpu)
    criterion = criterion.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    char_decoder = char_decoder.cuda()
    torch.cuda.empty_cache()
  else:
    gpu = False

  # Mixed Precision Setup
  if args.use_amp:
    print('Using Mixed Precision')
  grad_scaler = GradScaler(enabled=args.use_amp)

  # Initialize Checkpoint 
  if args.load_checkpoint:
    start_epoch, best_loss = load_checkpoint(encoder, decoder, optimizer, scheduler, args.checkpoint_path)
    print(f'Resuming training from checkpoint starting at epoch {start_epoch}.')
  else:
    start_epoch = 0
    best_loss = float('inf')

  # Train Loop
  optimizer.zero_grad()
  for epoch in range(start_epoch, args.epochs):
    torch.cuda.empty_cache()

    #variational noise for regularization
    add_model_noise(encoder, std=args.variational_noise_std, gpu=gpu)
    add_model_noise(decoder, std=args.variational_noise_std, gpu=gpu)

    # Train/Validation loops
    wer, loss = train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, args, gpu=gpu) 
    valid_wer, valid_loss = validate(encoder, decoder, char_decoder, criterion, test_loader, args, gpu=gpu)
    print(f'Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train Loss: {loss}')  

    # Save checkpoint 
    if valid_loss <= best_loss:
      print('Validation loss improved, saving checkpoint.')
      best_loss = valid_loss
      save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, args.checkpoint_path)

def train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, args, gpu=True):
  ''' Run a single training epoch '''

  wer = WordErrorRate()
  error_rate = AvgMeter()
  avg_loss = AvgMeter()
  text_transform = TextTransform()

  encoder.train()
  decoder.train()
  for i, batch in enumerate(train_loader):
    scheduler.step()
    gc.collect()
    spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
    
    # Move to GPU
    if gpu:
      spectrograms = spectrograms.cuda()
      labels = labels.cuda()
      input_lengths = torch.tensor(input_lengths).cuda()
      label_lengths = torch.tensor(label_lengths).cuda()
      mask = mask.cuda()
    
    # Update models
    with autocast(enabled=args.use_amp):
      outputs = encoder(spectrograms, mask)
      outputs = decoder(outputs)
      loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
    grad_scaler.scale(loss).backward()
    if (i+1) % args.accumulate_iters == 0:
      grad_scaler.step(optimizer)
      grad_scaler.update()
      optimizer.zero_grad()
    avg_loss.update(loss.detach().item())

    # Predict words, compute WER
    inds = char_decoder(outputs.detach())
    predictions = []
    for sample in inds:
      predictions.append(text_transform.int_to_text(sample))
    error_rate.update(wer(predictions, references) * 100)

    # Print metrics and predictions 
    if (i+1) % args.report_freq == 0:
      print(f'Step {i+1} - Avg WER: {error_rate.avg}%, Avg Loss: {avg_loss.avg}')   
      print('Sample Predictions: ', predictions)
    del spectrograms, labels, input_lengths, label_lengths, references, outputs, inds, predictions
  return error_rate.avg, avg_loss.avg

def validate(encoder, decoder, char_decoder, criterion, test_loader, args, gpu=True):
  ''' Evaluate model on test dataset. '''

  avg_loss = AvgMeter()
  error_rate = AvgMeter()
  wer = WordErrorRate()
  text_transform = TextTransform()

  encoder.eval()
  decoder.eval()
  for i, batch in enumerate(test_loader):
    gc.collect()
    spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
  
    # Move to GPU
    if gpu:
      spectrograms = spectrograms.cuda()
      labels = labels.cuda()
      input_lengths = torch.tensor(input_lengths).cuda()
      label_lengths = torch.tensor(label_lengths).cuda()
      mask = mask.cuda()

    with torch.no_grad():
      with autocast(enabled=args.use_amp):
        outputs = encoder(spectrograms, mask)
        outputs = decoder(outputs)
        loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
      avg_loss.update(loss.item())

      inds = char_decoder(outputs.detach())
      predictions = []
      for sample in inds:
        predictions.append(text_transform.int_to_text(sample))
      error_rate.update(wer(predictions, references) * 100)
  return error_rate.avg, avg_loss.avg


if __name__ == '__main__':
  main() 