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
    """
    
    """
    data = []
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                #cv default flag is color image. Hence, specify grayscale
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr,(img_size,img_size))
                data.append([resized_arr,class_num])
            except Exception as e:
                print(e)
    return data


def load_to_list(dataset_type,dataset_string,x_train,x_test,y_train,y_test):

    if dataset_string == "train":
        for image, label in dataset_type:
            x_train.append(image)
            y_train.append(label)
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
    # Normalize the data
    x_train = np.array(x_train) / 255
    x_test = np.array(x_test) / 255

    return x_train, x_test