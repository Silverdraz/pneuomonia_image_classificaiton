"""
A custom dataset class that can be extended to all image classification problems, even though
it is now a custom dataset for the x-ray images
"""

#Import statements
from torch.utils.data import Dataset #custom dataset 

class XRAYDatasetDataAug(Dataset):
    """Custom dataset class for data augmentation of x-ray images"""
    def __init__(self,data_df,transform):
        """Creates a custom dataset.

        Takes in the data for the dataset and transformations that will be performed
        when items are retrieved

        Args:
            data_df: data for the custom dataset
            transform: list of transforms for data augmentation
        """
        self.tensor_dataset = data_df
        self.transform = transform
    
    def __len__(self):
        """Need to overide this method as mentioned in docs."""
        number_rows = self.tensor_dataset.tensors[0].shape[0]
        return number_rows
    
    def __getitem__(self,index):
        """Need to overide this method as mentioned in docs."""
        image = self.tensor_dataset.tensors[0][index]
        label = self.tensor_dataset.tensors[1][index]
        # Apply online data augmentation
        if self.transform:
            image = self.transform(image)
        return (image,label)