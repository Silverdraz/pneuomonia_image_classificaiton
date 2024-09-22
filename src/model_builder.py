"""
Create various model architectures for training, model comparisons, evaluations and subsequently deployment of final model
"""

#torch import statements
from torch import nn #Basic Building Blocks for pytorch


class VGGNetBaseline(nn.Module):
    """ Replicating 1 block subset of VGGNet Architecture. A very tiny conv net for baseline benchmarking"""
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(401408),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.classifier(x)
        return x
    

class VGGNet3Block(nn.Module):
    """ Replicating 3 block subset of VGGNet Architecture to boost model's capacity for learning """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(401408),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.classifier(x)
        return x



class VGGNet3BlockBN(nn.Module):
    """ Replicating 3 block subset of VGGNet Architecture to boost model's capacity for learning """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(401408),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.classifier(x)
        return x



class VGGNetBN(nn.Module):
    """ Replicating subset of VGGNet Architecture. Including batch normalisation to the architecture """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
               
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(401408),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.classifier(x)
        return x
    

    