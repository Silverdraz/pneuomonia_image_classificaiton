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


class VGGNet2Block(nn.Module):
    """ Replicating 2 block subset of VGGNet Architecture to boost model's capacity for learning"""
    
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
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(401408),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
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
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x



class VGGNet3BlockBN(nn.Module):
    """ Replicating 3 block subset of VGGNet Architecture to address overfitting """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
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
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x


class VGGNet4Block(nn.Module):
    """ Replicating 4 block subset of VGGNet Architecture to check if increased model architecture can improve learning"""
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2))
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(100352),out_features=2)
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """ Helper Class acting as a residual block for custom resnet"""
    def __init__(self,channels):
        #super(ResidualBlock,self).__init__()
        super().__init__()
        
        self.block = nn.Sequential(
                nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[1],
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channels[1],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0),   
                nn.BatchNorm2d(channels[2])
        )

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(2, 2),
                                padding=0),
                nn.BatchNorm2d(channels[2])
        )
        
    def forward(self,x):
        shortcut = x
        
        block = self.block(x)
        shortcut = self.shortcut(x)    
        x = nn.functional.relu(block+shortcut)

        return x
        


class ResNet2Block(nn.Module):
    """ResNet custom class with a 2 block architecture"""
    def __init__(self):
        super().__init__()
        
        self.residual_block_1 = ResidualBlock(channels=[1,4,8])
        self.residual_block_2 = ResidualBlock(channels=[8,16,32])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(100352),out_features=2)
        )
        
    def forward(self,x):
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.classifier(x)
        return x
    

class ResNet3Block(nn.Module):
    """ResNet custom class with a 3 block architecture"""
    def __init__(self):
        super().__init__()
        
        self.residual_block_1 = ResidualBlock(channels=[1,4,8])
        self.residual_block_2 = ResidualBlock(channels=[8,16,32])
        self.residual_block_3 = ResidualBlock(channels=[32,64,128])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(100352),out_features=2)
        )
        
    def forward(self,x):
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.classifier(x)
        return x
      
class ResidualBlockReg(nn.Module):
    """ Helper Class acting as a residual block for custom resnet with extra regularisation (nn.Dropout2d) incorporated into the model"""
    def __init__(self,channels):
        #super(ResidualBlock,self).__init__()
        super().__init__()
        
        self.block = nn.Sequential(
                nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[1],
                                kernel_size=(3, 3),
                                stride=(2, 2),
                                padding=1),
                nn.BatchNorm2d(channels[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channels[1],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0),   
                nn.Dropout2d(p=0.3)
        )

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=channels[0],
                                out_channels=channels[2],
                                kernel_size=(1, 1),
                                stride=(2, 2),
                                padding=0),
                nn.BatchNorm2d(channels[2])
        )
        
    def forward(self,x):
        shortcut = x
        
        block = self.block(x)
        shortcut = self.shortcut(x)    
        x = nn.functional.relu(block+shortcut)

        return x
        


class ResNet3Reg(nn.Module):
    """ResNet custom class with a 3 block architecture with extra regularisation incorporated at the Residual Blocks and Sequential Blocks"""
    def __init__(self):
        super().__init__()
        
        self.residual_block_1 = ResidualBlockReg(channels=[1,4,8])
        self.residual_block_2 = ResidualBlockReg(channels=[8,16,32])
        self.residual_block_3 = ResidualBlockReg(channels=[32,64,128])
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(100352),out_features=128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=(128),out_features=2)
        )
        
    def forward(self,x):
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.classifier(x)
        return x
      
    
class Conv_Reg_1(nn.Module):
    """ Replicating a deeper conv net architecture with regularisation added by introducing BatchNorm2d and Dropouts """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2,2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=(2,2)),
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=12544,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128,out_features=2)
            
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x


class Conv_Reg_Recommended(nn.Module):
    """ Recommended deeper conv net architecture with regularisation added in the form of Batchnorm2d, Dropout2d. Best practices
        are adopted for this model by having batchnorm2d and dropout2d to be further away due to the variance shift during training
        and validation time. Dropout is located further away from the end of the classifier to prevent the model from being unable 
        to learn
    """
    
    def __init__(self):
        super().__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(0.2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout2d(p=0.2)
        )
     
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features=12544,out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=2)
            
        )
        
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x
