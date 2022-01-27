import torch as t

class GumbelSoftmax(t.nn.Module):
    def __init__(self, seq_len: int, vocab_size: int, eps: float = 1e-6):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.eps = eps

        self.weight = t.nn.Parameter(t.rand((seq_len, vocab_size)))

    
    def forward(self, batch_size, temp):
        gumbel_noise = -t.log(-t.log(t.rand((batch_size,) + self.weight.shape, device=self.weight.device)))
            
        return t.softmax((t.log(t.softmax(self.weight, dim=-1) + self.eps) + gumbel_noise) / temp, dim=-1)
