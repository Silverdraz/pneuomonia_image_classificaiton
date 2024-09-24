"""
Preprocess the images by resizing them to 224 x 224 size, followed by loading the images into list. The images are then converted to numpy 
and normalized
"""

#Import statements
import os #os
import cv2 #comp vision library
import numpy as np #array/matrix manipulation

# Set Data Path to data directory
DATA_PATH = r"..\data\raw"

#0 is Pneuomonia positive, 1 is normal
labels = ['PNEUMONIA', 'NORMAL']
#resize to standard image size
img_size = 224

def get_training_data(data_dir):
    """Load the images from data_dir (train/val/test) and attach the classification label to the respective examples

    The images are resized to a standard size (img_size) of 224 x 224 These
    are then converted to grayscale

    Args:
        data_dir: 

    Returns:
        data: list of arrays that represent the different images and their respective class labels
    """
    #Initialise empty data list to append the individual images and class labels
    data = []
    #train/val/test dataset are split into NORMAL & PNEUMONIA classes. Hence it requires this iteration
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                #cv default flag is color image. Hence, specify grayscale
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                #Resize the image to img_size
                resized_arr = cv2.resize(img_arr,(img_size,img_size))
                #Append both the array (image) and the clas label
                data.append([resized_arr,class_num])
            except Exception as e:
                print(e)
    return data


def load_to_list(dataset_type,dataset_string,x_train,x_test,y_train,y_test):
    """Load the returned data that are clustered as [img_array,label] so that it can be separated into a list of arrays(images) and it's
    into a list of arrays(images) and it's respective class labels in another list for training/validation

    Args:
        dataset_type: train/val/test data that is returned from get_training_data with data represented as [img_arr,class_num]
        dataset_string: str to indicate whether to add to x_train or x_test
        x_train: list for img arrays in train/val directory
        x_test: list for img_arrays in test directory
        y_train: list for labels of img_arrays in train/val directory
        y_test: list for labels of img_arrays in test directory

    Returns:
        x_train: list for img arrays in train/val directory
        x_test: list for img_arrays in test directory
        y_train: list for labels of img_arrays in train/val directory
        y_test: list for labels of img_arrays in test directory
    """

    #Check if the datasets are comning from train/val/test data directory
    if dataset_string == "train":
        for image, label in dataset_type:
            x_train.append(image)
            y_train.append(label)
    #Val dataset is included into x_train for stratified splitting since it only has 8 samples (too small)
    elif dataset_string == "val":
        for image, label in dataset_type:
            x_train.append(image)
            y_train.append(label)
    elif dataset_string == "test":
        for image, label in dataset_type:
            x_test.append(image)
            y_test.append(label)

    return x_train,x_test,y_train,y_test

def array_normalize_data(x_train,x_test):
    """Normalise the arrays to 0 - 1 for faster computation and 
    Args:
        x_train: list of arrays for the various train/val images
        x_test: list of arrays for the various test images 

    Returns:
        x_train: list of arrays for the various train/val images
        x_test : list of arrays for the various test images
    """

    # Normalize the data to (0-1) range
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255

    return x_train, x_test