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

        self.register_buffer('pe', pe)

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




    
