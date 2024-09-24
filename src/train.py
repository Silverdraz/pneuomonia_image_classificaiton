"""
Main file for preparing the data required for model training. Models are loaded from model_builder.py with main training process completed
using functions from engine.py. Model evaluations are then performed across the models by using the results saved as model_results.csv
followed by the loss curves and accuracy curves that are saved in the visualisations
"""



#import statements
import os #os 
import numpy as np #array/matrix manipulation
from sklearn.model_selection import train_test_split # Split data into train and test sets
import pandas as pd #dataframe manipulation

#Torch import statements
import torch #General torch library
from torch.utils.data import DataLoader #Mini batch data loader
from torch import nn #Basic Building Blocks for pytorch
from torchvision import transforms #Data Augmentation
import torchvision.models as models #model zoo

#import modules
import loading_preprocessing_data #data preprocessing and loading module
import model_builder #model architectures
import engine #training of model
import dataset #custom dataset module
import plots #custom plots viz module


# Set Data Path to data directory
DATA_PATH = r"..\data\raw"
#Set path to save the loss curves and f1 socres
SAVE_VIS_PATH = r"..\visualisations"
RANDOM_SEED = 42 #Set random seeed for reproducibility
batch_size = 16 # Batch Size hyperparameter for data loader


def run_vggnet_baseline(train_dataloader,
                        val_dataloader,
                        NUM_EPOCHS,
                        device):
    """Trains a one block custom vggnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
    torch.manual_seed(42)
    model = model_builder.VGGNetBaseline().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)
    

    # Start training with help from engine.py
    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"vggnet")
    return results

def run_vggnet_2block(train_dataloader,
                      val_dataloader,
                      NUM_EPOCHS,
                      device):
    """Trains a 2 block custom vggnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """


    torch.manual_seed(42)
    model_1 = model_builder.VGGNet2Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_1,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    plots.plot_loss_acc_curves(results,"vggnet_2block")
    return results


def run_vggnet_3block(train_dataloader,
                      val_dataloader,
                      NUM_EPOCHS,
                      device):
    
    """Trains a 3 block custom vggnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
     
    torch.manual_seed(42)
    model_2 = model_builder.VGGNet3Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_2,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"vggnet_3block")
    return results

def run_vggnet_3blockbn(train_dataloader,
                        val_dataloader,
                        NUM_EPOCHS,
                        device):
    
    """Trains a 3 block custom vggnet with batch normalisation and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
    torch.manual_seed(42)
    model_3 = model_builder.VGGNet3BlockBN().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_3.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_3,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    plots.plot_loss_acc_curves(results,"vggnet_3blockbn")
    return results

def run_vggnet_3block_aug(train_dataloader,
                          val_dataloader,
                          NUM_EPOCHS,
                          device):
    
    """Trains a 3 block custom vggnet with data aug and batch norm and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
    torch.manual_seed(42)

    #Batch Size hyperparameter
    batch_size = 16

    #train_tensors = torch.utils.data.TensorDataset(X_train,y_train)
    #val_tensors = torch.utils.data.TensorDataset(X_val,y_val)
    train_transform = transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=2)])

    # Turn dataset into iterables of mini batches
    #Shuffle is true for training to prevent learning of spurious correlation or noise
    train_aug_dataset = dataset.XRAYDatasetDataAug(train_tensors,train_transform)
    train_dataloader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True)

    val_aug_dataset = dataset.XRAYDatasetDataAug(val_tensors,None)
    val_dataloader = DataLoader(val_aug_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataloaders: {train_dataloader,val_dataloader}")
    print(f"Length of train dataloaders: {len(train_dataloader)} of batch size {batch_size}")
    print(f"Length of validation dataloaders: {len(val_dataloader)} of batch size {batch_size}")


    model_4 = model_builder.VGGNet3BlockBN().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_4.parameters(),lr=0.01)
    

    # Start training with help from engine.py
    results = engine.train(model=model_4,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"vggnet_3block_aug")
    return results


def run_vggnet_4block(train_dataloader,
                      val_dataloader,
                      NUM_EPOCHS,
                      device):
    
    """Trains a 4 block custom vggnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
    torch.manual_seed(42)
    model_5 = model_builder.VGGNet4Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_5.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_5,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)

    plots.plot_loss_acc_curves(results,"vggnet_4block")
    return results


def run_resnet_2block(train_dataloader,
                      val_dataloader,
                      NUM_EPOCHS,
                      device):
    """Trains a 2 block custom resnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """    

    torch.manual_seed(42)
    model_6 = model_builder.ResNet2Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_6.parameters(),lr=0.01)
    
    print(device,"this is device")

    # Start training with help from engine.py
    results = engine.train(model=model_6,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"resnet_2block")
    return results

def run_resnet_3block(train_dataloader,
                      val_dataloader,
                      NUM_EPOCHS,
                      device):
    """Trains a 3 block custom resnet and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """    

    torch.manual_seed(42)
    model_7 = model_builder.ResNet3Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_7.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_7,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"resnet_3block")
    return results

def run_resnet_3blockreg(train_dataloader,
                        val_dataloader,
                        NUM_EPOCHS,
                        device):
    """Trains a 3 block custom resnet with regularisaiton added and plot the results

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """
    torch.manual_seed(42)
    model_8 = model_builder.ResNet3Reg().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_8.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_8,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"resnet_3block_reg")
    return results

def run_conv_reg_1(train_dataloader,
                   val_dataloader,
                   NUM_EPOCHS,
                   device):
    
    """Trains a deep conv net with regularisatino (dropout and batchnorm) included

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """    

    torch.manual_seed(42)
    model_9 = model_builder.Conv_Reg_1().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_9.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_9,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"conv_reg_1")
    return results

def run_pretrained_resnet(train_dataloader,
                          val_dataloader,
                          NUM_EPOCHS,
                          device):
    
    """Finetunes the pretrained resnet model. Resnet is chosen over vggnet as it has 18 layers and may have exploding
        and vanishing gradients. 

    Args:
    train_dataloader: Batch load the train data
    val_dataloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """    

    torch.manual_seed(42)

    #Use the normalize values to be consistent with the pretrained normalization
    normalize = transforms.Normalize(mean=[(0.485+0.456+0.406)/3.0],
                                     std=[(0.229+0.224+0.225)/3.0])

    #Have to be in range 0-1(already) and at least 224 x 224(already) and use above normalization
    train_transform = transforms.Compose([normalize])    
    val_transform = transforms.Compose([normalize])

    # Turn dataset into iterables of mini batches
    #Shuffle is true for training to prevent learning of spurious correlation or noise
    resnet_pretrain_train_dataset = dataset.XRAYDatasetDataAug(train_tensors,train_transform)
    train_dataloader = DataLoader(resnet_pretrain_train_dataset, batch_size=batch_size, shuffle=True)


    resnet_pretrain_val_dataset = dataset.XRAYDatasetDataAug(val_tensors,val_transform)
    val_dataloader = DataLoader(resnet_pretrain_val_dataset, batch_size=batch_size, shuffle=False)

    model_10 = models.resnet18(pretrained=True).to(device)

    model_10 = freeze_open_layers(model_10)

    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_10.parameters(),lr=0.01)
    

    # Start training with help from engine.py
    results = engine.train(model=model_10,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"finetuned_resnet18")
    return results


def freeze_open_layers(model_10):
    #Freeze the layers. Replace first layer conv1 to accept a single channel image instead of 3 channels
    model_10.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)

    #freeze the trained parameters
    for param in model_10.parameters():
        param.requires_grad = False

    #Replace default num_classes to 10 and set this Linear layer to trainable
    model_10.fc = nn.Linear(512,2)
    model_10.fc.requires_grad = True

    return model_10



def run_conv_reg_rec(train_dataloader,
                   val_dataloader,
                   NUM_EPOCHS,
                   device):
    """Trains a deep conv net with regularisation added (batchnorm and dropout) with optimal placement of these layers as
        batchnorm and dropout is recommended to be placed further apart.

    Argsdataloader: Batch load the train data
    val_da:
    train_taloader: Batch load
    NUM_EPOCHS: Number of epochs to run for
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    results: a dictionary consisting of keys such as train_loss_list, train_accuracy_list, train_f1_list .... for validation as well
    """    

    torch.manual_seed(42)
    model_11 = model_builder.Conv_Reg_Recommended().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_11.parameters(),lr=0.01)

    # Start training with help from engine.py
    results = engine.train(model=model_11,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"conv_reg_rec")
    return results

def load_train_data():
    """Returns the x_train, x_test, y_train, y_test all separated into their respective lists

    Returns:
        x_train, x_test, y_train, y_test
    """
    train = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"train"))
    test = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"test"))
    val = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"val"))

    x_train, x_test, y_train, y_test = [], [], [], []
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(train,"train",x_train,x_test,y_train,y_test)
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(val,"val",x_train,x_test,y_train,y_test)
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(test,"test",x_train,x_test,y_train,y_test)
    
    return x_train,x_test, y_train, y_test
   

def convert_to_tensors(x_train,y_train,x_test,y_test):
    """Convert from numpy arrays to pytorch tensors for training

    Args:
        x_train: numpy array of train images
        y_train: numpy array of class labels
        x_test: numpy array of train images
        y_test: numpy array of class labels

    Returns:
        x_train, y_train, x_test,y_test (in pytorch tensor format)
    """
    #Convert to tensors in particular long tensor for y_train since nn.crossentropy requires y labels to
    #be long type
    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    x_test = torch.from_numpy(x_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    return x_train, y_train, x_test, y_test

def results_for_model(results,
                      train_loss_list,
                      val_loss_list,
                      train_accuracy_list,
                      val_accuracy_list,
                      train_f1_list,
                      val_f1_list):
    """Save the last epoch results into the list as a dataframe

    Args:
        results: dictionary consisting of the list of train_loss_list, val_loss_list.... for accuracy and f1 score
        train_loss_list: train loss results across epochs
        val_loss_list: val loss results across
        train_accuracy_list: train accuracy results across epochs
        val_accuracy_list: val accuracy results across epochs
        train_f1_list: train f1 score results across epochs
        val_f1_list: val f1 score results across epochs
      
    Returns:
        train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list
    """
    #Extract out the results in the last epoch 
    train_loss, val_loss = results["train_loss_list"][-1], results["test_loss_list"][-1]
    train_accuracy, val_accuracy = results["train_accuracy_list"][-1], results["test_accuracy_list"][-1]
    train_f1, val_f1 = results["train_f1_list"][-1], results["test_f1_list"][-1], 
    #Append the respective results into the list
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_accuracy_list.append(train_accuracy)
    val_accuracy_list.append(val_accuracy)
    train_f1_list.append(train_f1)
    val_f1_list.append(val_f1)

    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list


if __name__ == "__main__":

    #Retrieve data in x (arrays of images) and y (class labels)
    x_train, x_test, y_train, y_test = load_train_data()

    #Normalised arrays of images
    x_train, x_test = loading_preprocessing_data.array_normalize_data(x_train,x_test)

    # resize data for deep learning 
    img_size = 224
    x_train = x_train.reshape(-1,1,img_size,img_size)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1,1,img_size, img_size)
    y_test = np.array(y_test)

    #Convert to tensors in particular long tensor for y_train since nn.crossentropy requires y labels to
    #be long type
    x_train, y_train, x_test, y_test = convert_to_tensors(x_train,y_train,x_test,y_test)

    #Split the dataset into train and validation

    #Split into x_train and y_train using stratify since the dataset is imbalanced
    x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                        y_train, 
                                                        test_size=0.15, # 15% test, 85% train,
                                                        stratify=y_train,
                                                        random_state=RANDOM_SEED) # make the random split reproducible


    print(x_val.shape,"this is the shape of x val")
    train_tensors = torch.utils.data.TensorDataset(x_train,y_train)
    val_tensors = torch.utils.data.TensorDataset(x_val,y_val)

    # Turn dataset into iterables of mini batches
    #Shuffle is true for training to prevent learning of spurious correlation or noise
    train_dataloader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tensors, batch_size=batch_size, shuffle=False)

    print(f"Dataloaders: {train_dataloader,val_dataloader}")
    print(f"Length of train dataloaders: {len(train_dataloader)} of batch size {batch_size}")
    print(f"Length of validation dataloaders: {len(val_dataloader)} of batch size {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 10

    #Initialise the list to retrieve the evaluation scores of the last epoch for every model
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = [], [], [], [], [], []
    classifier_names = ["vggnet_baseline","vggnet_2block","vggnet_3block","vggnet_3blockbn","vggnet_3block_aug","vggnet_4block",
                        "resnet_2block","resnet_3block","finetuned_resnet","resnet_3block_reg","run_conv_reg_1","run_conv_reg_rec"]
    
    #Instantiate an instance of the model from the "model_builder.py" script
    results = run_vggnet_baseline(train_dataloader,
                                    val_dataloader,
                                    NUM_EPOCHS,
                                    device)
 
    # Can use "\" for better formatting if needed
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    #################################

    results = run_vggnet_2block(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    # #################################
    results = run_vggnet_3block(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)

    ################################
    
    results = run_vggnet_3blockbn(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)

    # # ##############################
    results = run_vggnet_3block_aug(train_dataloader,
                                    val_dataloader,
                                    NUM_EPOCHS,
                                    device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    
    # # ############################
    results = run_vggnet_4block(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    # # ############################
    results = run_resnet_2block(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    # # # #############################
    results = run_resnet_3block(train_dataloader,
                                val_dataloader,
                                NUM_EPOCHS,
                                device) 
       
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    
    
    # ##############################
    results = run_pretrained_resnet(train_dataloader,
                                    val_dataloader,
                                    NUM_EPOCHS,
                                    device)

    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    
    # ############################
    results = run_resnet_3blockreg(train_dataloader,
                                    val_dataloader,
                                    NUM_EPOCHS,
                                    device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)
    ############################
    results = run_conv_reg_1(train_dataloader,
                            val_dataloader,
                            NUM_EPOCHS,
                            device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)

    ##########################
    results = run_conv_reg_rec(train_dataloader,
                            val_dataloader,
                            NUM_EPOCHS,
                            device)
    
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, train_f1_list, val_f1_list = results_for_model(results,train_loss_list,
                                                                                                                           val_loss_list,train_accuracy_list,
                                                                                                                           val_accuracy_list,train_f1_list,
                                                                                                                           val_f1_list)

    
    results_df = pd.DataFrame({'train_loss':train_loss_list,'val loss':val_loss_list,
                               'train accuracy':train_accuracy_list,'val accuracy': val_accuracy_list,
                               "train_f1":train_f1_list, "val f1":val_f1_list},index=classifier_names)
    results_df.to_csv("model_results.csv")