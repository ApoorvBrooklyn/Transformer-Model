import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PostionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Creating a Matrix of Shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #Create a vector os shape (seqq_len, 1)
        position = torch.arrange(0, seq_len, dtype = torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model)) # We calculate in Log space for numerical stability

        # Apply Sin to even position
        pe[:,0::2] = torch.sin(position*div_term)

        #Apply cos to odd position
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # We add a another dimension. So PE is of dimension (1, Seq_Len, d_model)

        self.register_buffer('pe', pe) # Tensor would be saved along with the model. When you dont wanna 

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied Factor
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim=-1, keepdim = True)
        return self.alpha * (x - mean)/ (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.Linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.droput = nn.Dropout(dropout)
        self.Linear_2 = nn.Linear(d_ff, d_model) #W2 and B2

    def forward(self,x):
        #(Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.Linear_2(self.droput(torch.relu(self.Linear_1(x))))
    

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout: float) -> None: # h represents number of head
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropuout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # Wq query
        self.w_k = nn.Linear(d_model, d_model) # Wk Key
        self.w_v = nn.Linear(d_model, d_model) # Wv Value

        self.w_o = nn.Linear(d_model, d_model) # Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_Scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_Scores.masked_fill_(mask == 0, -1e9)
        attention_Scores = attention_Scores.softmax(dim = -1)
        if dropout is not None:
            attention_Scores = dropout(attention_Scores)

        return (attention_Scores @ value), attention_Scores
        
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v) # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        ## (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_Scores = MultiheadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def init__(self, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        #self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attetion_block: MultiheadAttentionBlock, cross_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attetion_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range (3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output, src_mask, tgt_mask )
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super.__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #(batch, seq_len, d_model) -> (Batch, SeqLen, Vocab_size)
        return torch.log_softmax(self.proj(x), dim = 1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed:InputEmbeddings, src_pos: PostionalEncoding, tgt_pos: PostionalEncoding, projection_layer: ProjectionLayer) -> None:
        