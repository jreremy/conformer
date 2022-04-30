import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
  '''
    Generate positional encodings used in the relative multi-head attention module.
    These encodings are the same as the original transformer model: https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve
    
    Outputs
      Tensor (len, d_model): Positional encodings
  '''
  def __init__(self, d_model, max_len=10000):
    super(PositionalEncoder, self).__init__()
    self.d_model = d_model
    encodings = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float)
    inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
    encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
    encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
    self.register_buffer('encodings', encodings)
    
  def forward(self, len):
      return self.encodings[:len, :]

class RelativeMultiHeadAttention(nn.Module):
  '''
    Relative Multi-Head Self-Attention Module. 
    Method proposed in Transformer-XL paper: https://arxiv.org/abs/1901.02860

    Parameters:
      d_model (int): Dimension of the model
      num_heads (int): Number of heads to split inputs into
      dropout (float): Dropout probability
      positional_encoder (nn.Module): PositionalEncoder module
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the attention module.
  
  '''
  def __init__(self, d_model=144, num_heads=4, dropout=0.1, positional_encoder=PositionalEncoder(144)):
    super(RelativeMultiHeadAttention, self).__init__()

    #dimensions
    assert d_model % num_heads == 0
    self.d_model = d_model
    self.d_head = d_model // num_heads
    self.num_heads = num_heads

    # Linear projection weights
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_pos = nn.Linear(d_model, d_model, bias=False)
    self.W_out = nn.Linear(d_model, d_model)

    # Trainable bias parameters
    self.u = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    self.v = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
    torch.nn.init.xavier_uniform_(self.u)
    torch.nn.init.xavier_uniform_(self.v)

    # etc
    self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
    self.positional_encoder = positional_encoder
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    batch_size, seq_length, _ = x.size()

    #layer norm and pos embeddings
    x = self.layer_norm(x)
    pos_emb = self.positional_encoder(seq_length)
    pos_emb = pos_emb.repeat(batch_size, 1, 1)

    #Linear projections, split into heads
    q = self.W_q(x).view(batch_size, seq_length, self.num_heads, self.d_head)
    k = self.W_k(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    v = self.W_v(x).view(batch_size, seq_length, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)
    pos_emb = self.W_pos(pos_emb).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1) # (batch_size, num_heads, d_head, time)

    #Compute attention scores with relative position embeddings
    AC = torch.matmul((q + self.u).transpose(1, 2), k)
    BD = torch.matmul((q + self.v).transpose(1, 2), pos_emb)
    BD = self.rel_shift(BD)
    attn = (AC + BD) / math.sqrt(self.d_model)

    #Mask before softmax with large negative number
    if mask is not None:
      mask = mask.unsqueeze(1)
      mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
      attn.masked_fill_(mask, mask_value)

    #Softmax
    attn = F.softmax(attn, -1)

    #Construct outputs from values
    output = torch.matmul(attn, v.transpose(2, 3)).transpose(1, 2) # (batch_size, time, num_heads, d_head)
    output = output.contiguous().view(batch_size, -1, self.d_model) # (batch_size, time, d_model)

    #Output projections and dropout
    output = self.W_out(output)
    return self.dropout(output)


  def rel_shift(self, emb):
    '''
      Pad and shift form relative positional encodings. 
      Taken from Transformer-XL implementation: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py 
    '''
    batch_size, num_heads, seq_length1, seq_length2 = emb.size()
    zeros = emb.new_zeros(batch_size, num_heads, seq_length1, 1)
    padded_emb = torch.cat([zeros, emb], dim=-1)
    padded_emb = padded_emb.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
    shifted_emb = padded_emb[:, :, 1:].view_as(emb)
    return shifted_emb


class ConvBlock(nn.Module):
  '''
    Conformer convolutional block.

    Parameters:
      d_model (int): Dimension of the model
      kernel_size (int): Size of kernel to use for depthwise convolution
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask: Unused
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the convolution module
  
  '''
  def __init__(self, d_model=144, kernel_size=31, dropout=0.1):
    super(ConvBlock, self).__init__()
    self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)
    kernel_size=31
    self.module = nn.Sequential(
      nn.Conv1d(in_channels=d_model, out_channels=d_model * 2, kernel_size=1), # first pointwise with 2x expansion
      nn.GLU(dim=1),
      nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding='same', groups=d_model), # depthwise
      nn.BatchNorm1d(d_model, eps=6.1e-5),
      nn.SiLU(), # swish activation
      nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1), # second pointwise
      nn.Dropout(dropout)
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.transpose(1, 2) # (batch_size, d_model, seq_len)
    x = self.module(x)
    return x.transpose(1, 2)

class FeedForwardBlock(nn.Module):
  '''
    Conformer feed-forward block.

    Parameters:
      d_model (int): Dimension of the model
      expansion (int): Expansion factor for first linear layer
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask: Unused
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the feed-forward module
  
  '''
  def __init__(self, d_model=144, expansion=4, dropout=0.1):
    super(FeedForwardBlock, self).__init__()
    self.module = nn.Sequential(
      nn.LayerNorm(d_model, eps=6.1e-5),
      nn.Linear(d_model, d_model * expansion), # expand to d_model * expansion
      nn.SiLU(), # swish activation
      nn.Dropout(dropout),
      nn.Linear(d_model * expansion, d_model), # project back to d_model
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.module(x)

class Conv2dSubsampling(nn.Module):
  '''
    2d Convolutional subsampling. 
    Subsamples time and freq domains of input spectrograms by a factor of 4, d_model times. 

    Parameters:
      d_model (int): Dimension of the model
    
    Inputs:
      x (Tensor): Input spectrogram (batch_size, time, d_input)
    
    Outputs:
      Tensor (batch_size, time, d_model * (d_input // 4)): Output tensor from the conlutional subsampling module
  
  '''
  def __init__(self, d_model=144):
    super(Conv2dSubsampling, self).__init__()
    self.module = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2),
      nn.ReLU(),
    )

  def forward(self, x):
    output = self.module(x.unsqueeze(1)) # (batch_size, 1, time, d_input)
    batch_size, d_model, subsampled_time, subsampled_freq = output.size()
    output = output.permute(0, 2, 1, 3)
    output = output.contiguous().view(batch_size, subsampled_time, d_model * subsampled_freq)
    return output

class ConformerBlock(nn.Module):
  '''
    Conformer Encoder Block. 

    Parameters:
      d_model (int): Dimension of the model
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention
      positional_encoder (nn.Module): PositionalEncoder module
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): (batch_size, time, d_model)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the conformer block.
  
  '''
  def __init__(
          self,
          d_model=144,
          conv_kernel_size=31,
          feed_forward_residual_factor=.5,
          feed_forward_expansion_factor=4,
          num_heads=4,
          positional_encoder=PositionalEncoder(144),
          dropout=0.1,
  ):
    super(ConformerBlock, self).__init__()
    self.residual_factor = feed_forward_residual_factor
    self.ff1 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
    self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout, positional_encoder)
    self.conv_block = ConvBlock(d_model, conv_kernel_size, dropout)
    self.ff2 = FeedForwardBlock(d_model, feed_forward_expansion_factor, dropout)
    self.layer_norm = nn.LayerNorm(d_model, eps=6.1e-5)

  def forward(self, x, mask=None):
    x = x + (self.residual_factor * self.ff1(x))
    x = x + self.attention(x, mask=mask)
    x = x + self.conv_block(x)
    x = x + (self.residual_factor * self.ff2(x))
    return self.layer_norm(x)


class ConformerEncoder(nn.Module):
  '''
    Conformer Encoder Module. 

    Parameters:
      d_input (int): Dimension of the input
      d_model (int): Dimension of the model
      num_layers (int): Number of conformer blocks to use in the encoder
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention
      dropout (float): Dropout probability
    
    Inputs:
      x (Tensor): input spectrogram of dimension (batch_size, time, d_input)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices
    
    Outputs:
      Tensor (batch_size, time, d_model): Output tensor from the conformer encoder

  
  '''
  def __init__(
          self,
          d_input=80,
          d_model=144,
          num_layers=16,
          conv_kernel_size=31, 
          feed_forward_residual_factor=.5,
          feed_forward_expansion_factor=4,
          num_heads=4,
          dropout=.1,
  ):
    super(ConformerEncoder, self).__init__()
    self.conv_subsample = Conv2dSubsampling(d_model=d_model)
    self.linear_proj = nn.Linear(d_model * (((d_input - 1) // 2 - 1) // 2), d_model) # project subsamples to d_model
    self.dropout = nn.Dropout(p=dropout)
    
    # define global positional encoder to limit model parameters
    positional_encoder = PositionalEncoder(d_model) 
    self.layers = nn.ModuleList([ConformerBlock(
            d_model=d_model,
            conv_kernel_size=conv_kernel_size, 
            feed_forward_residual_factor=feed_forward_residual_factor,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
            dropout=dropout,
        ) for _ in range(num_layers)])

  def forward(self, x, mask=None):
    x = self.conv_subsample(x)
    if mask is not None:
      mask = mask[:, :-2:2, :-2:2] #account for subsampling
      mask = mask[:, :-2:2, :-2:2] #account for subsampling
      assert mask.shape[1] == x.shape[1], f'{mask.shape} {x.shape}'
    
    x = self.linear_proj(x)
    x = self.dropout(x)
    
    for layer in self.layers:
      x = layer(x, mask=mask)
    
    return x


class LSTMDecoder(nn.Module):
  '''
    LSTM Decoder

    Parameters:
      d_encoder (int): Output dimension of the encoder
      d_decoder (int): Hidden dimension of the decoder
      num_layers (int): Number of LSTM layers to use in the decoder
      num_classes (int): Number of output classes to predict
    
    Inputs:
      x (Tensor): (batch_size, time, d_encoder)
    
    Outputs:
      Tensor (batch_size, time, num_classes): Class prediction logits
  
  '''
  def __init__(self, d_encoder=144, d_decoder=320, num_layers=1, num_classes=29):
    super(LSTMDecoder, self).__init__()
    self.lstm = nn.LSTM(input_size=d_encoder, hidden_size=d_decoder, num_layers=num_layers, batch_first=True)
    self.linear = nn.Linear(d_decoder, num_classes)

  def forward(self, x):
    x, _ = self.lstm(x)
    logits = self.linear(x)
    return logits
