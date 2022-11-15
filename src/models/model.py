import os 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticsModelPytorch(nn.Module):

    def __init__(self, vocab_size, emb_dim, hid_dim, n_out, reduce):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding.
        @param n_out: size of the class.
        @param reduce: interaction type: [sum, concat, element-wise].
        """
        super().__init__()
        
        self.embedhypo = nn.Embedding(vocab_size, emb_dim, padding_idx = 1)

        self.embedprem = nn.Embedding(vocab_size, emb_dim, padding_idx = 1)

        self.linear1 = nn.Linear(emb_dim, hid_dim)
        
        self.linear2 = nn.Linear(hid_dim, n_out)

        self.linearconcat = nn.Linear(emb_dim*2, hid_dim)

        self.reduce = reduce

    def forward(self, data_hypo, length_hypo, data_prem, length_prem):
        """
            @param data_hypo: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
            @param length_hypo: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in data_hypo.
            @param data_prem: matrix of size (batch_size, max_sentence_length).
            @param length_hypo: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
                length of each sentences in data_prem.
            """
        
        # word embedding
        # combine to sentence
        data_hypo = torch.mean(self.embedhypo(data_hypo), dim = 1)
        data_prem = torch.mean(self.embedprem(data_prem), dim = 1)
        
        # concat (This will change embedding dimension, 2 times as many as before)
        if self.reduce == 'concat':
            x = torch.concat((data_prem, data_hypo), dim = -1)
            x = F.relu(self.linearconcat(x))
            x = self.linear2(x)

        else:
            # sum
            if self.reduce == 'sum':
                x = data_prem + data_hypo
                print(x.shape)

            # Hadamard (element-wise) product
            if self.reduce == 'hadamard':
                x = data_prem * data_hypo
            
          # hidden layers 
            
            x = F.relu(self.linear1(x))
            x = self.linear2(x)

        return x