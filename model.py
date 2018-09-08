import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Store these variables.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        # Embedding layer that turns words into a vector of a specified size.
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        captions = captions[:, :-1]  # For some reason, we strip the <end> token...
        embedded_input = self.word_embeddings(captions)
        
        # Now we need to add the features vector fist
        # We do this so the dimensions of features and embedded_input match and they can be concatenated.
        batched_features = features.unsqueeze(1)  
        embedded_input = torch.cat((batched_features, embedded_input), 1)
        x, _ = self.lstm(embedded_input)
        
        x = x.contiguous().view(x.size()[0]*x.size()[1], self.hidden_size)
        
        x = self.linear(x)
        
        return x

    def sample(self, features, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        inputs = features
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_id = predicted.item()
            sampled_ids.append(sampled_id)
            
            inputs = self.word_embeddings(predicted).unsqueeze(1)
            
            
        return sampled_ids
            