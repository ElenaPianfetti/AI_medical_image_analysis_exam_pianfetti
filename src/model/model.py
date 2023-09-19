import torch
import torch.nn as nn

from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights


class Adapter4_3(nn.Module):
    ''' Adapter then encoder'''

    def __init__(self, nn_channels, nn_classes=2):
        super(Adapter4_3, self).__init__()

        self.nn_channels = nn_channels
        self.nn_classes = nn_classes
        
        self.embedding_dim = 0

        # Initialize the adapter
        self.adapter = nn.Conv2d(in_channels = 4, out_channels = 3, kernel_size = 1)

        # Initialize the encoder
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        
        # remove the last layer from the encoder
        encoder = efficientnet_v2_s(weights=weights)
        
        self.embedding_dim += encoder.classifier[1].in_features
        
        # set the encoder.classifier to the identity
        encoder.classifier = nn.Identity()
        setattr(self, 'encoder', encoder)
        
        self.classifier = nn.Linear(self.embedding_dim, nn_classes)


    def forward(self, x):
        ''' Forward pass of the model '''

        embedding = self.embed(x)        
        x = self.classifier(embedding)
        
        return x
    
    def embed(self, x):
        ''' Forward pass of the model '''
        
        embedding = self.adapter(x)
        embedding = self.encoder(embedding)

        return embedding

class Encoder_x1(nn.Module):
    ''' Encoder for a single channel'''

    def __init__(self, nn_classes=2):
        super(Encoder_x1, self).__init__()
        self.nn_classes = nn_classes
        self.embedding_dim = 0

        # Initialize the adapter
        self.adapter = nn.Conv2d(in_channels = 4, out_channels = 3, kernel_size = 1)

        # Initialize encoder
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        # remove the last layer from the encoder
        encoder = efficientnet_v2_s(weights=weights)
        
        self.embedding_dim += encoder.classifier[1].in_features
        
        # set the encoder.classifier to the identity
        encoder.classifier = nn.Identity()
        setattr(self, f'encoder', encoder)
        
        self.classifier = nn.Linear(self.embedding_dim, nn_classes)


    def forward(self, x):
        ''' Forward pass of the model '''
        embedding = self.embed(x)
        x = self.classifier(embedding)
        return x
    
    def embed(self, x):
        ''' Forward pass of the model '''
        
        x = x.unsqueeze(1)
        # repeat the channel 4 times to match the input size of the encoder
        x = x.repeat(1, 4, 1, 1)
        embedding = self.adapter(x)
        embedding = self.encoder(embedding)

        return embedding