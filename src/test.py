"""
Main file for fitting the final model, saving the final model and estimating the final model's performance on the test set
with a confusion matrix
"""


#import statements
import os #os 
import numpy as np #array/matrix manipulation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #evaluation metric
import matplotlib.pyplot as plt


#Torch import statements
import torch #General torch library
from torch import nn #Basic Building Blocks for pytorch
from torch.utils.data import DataLoader

#import modules
import loading_preprocessing_data #data preprocessing and loading module
import model_builder #model architectures
import engine #training of model
import plots #custom plots viz module


# Set Data Path to data directory
DATA_PATH = r"..\data\raw"
SAVE_VIS_PATH = r"..\visualisations" #Path to visualisations
SAVE_MODEL_PATH = r"..\saved_model"
LOAD_MODEL_PATH = r"..\saved_model\best_model.pt" #path to best model

def final_model_run(train_dataloader,
                    test_dataloader,
                    NUM_EPOCHS,
                    device):
    """Trains the selected model from model comparison and evaluation on the whole train and validation dataset.
    Perform inference on test dataset to allow for a one-time/2-time evaluation on the test dataset to check generalisability

    Args:
        train_dataloader: Batch loader for train dataset
        test_dataloader: Batch loader for test dataset
        NUM_EPOCHS: Number of epochs to run for
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        final_model: the final model trained on train + validation datasset
        optimizer: optimizer parameters that have been used in the training
    """    
    torch.manual_seed(42)
    final_model = model_builder.VGGNet3BlockBN().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=final_model.parameters(),lr=0.01)

    print(device,"this is device")

    # Start training with help from engine.py
    results = engine.train(model=final_model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=NUM_EPOCHS,
                device=device)
    
    plots.plot_loss_acc_curves(results,"final_model")
    return final_model, optimizer

def evaluate_test_model(x_test,y_test):
    """Evaluate the final model performance on the test dataset (saved form best_model.pt) and
    plotted using a confusion matrix

    Args:
        x_test: list of test array images
        y_test: List of labels

    """
    #Load classifier (not fine-tuned weights)
    inference_model = model_builder.VGGNet3BlockBN()
    #Load the fine-tuned weights on chest-xray dataset
    inference_model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    #Set the mode to eval for inference (especially for dropout and batchnorm2d)
    inference_model.eval()

    #Inference/Prediction mode
    with torch.no_grad():
        logits = inference_model(x_test)
        #Retrieve the class label with highest softmax probability socre
        y_pred = torch.softmax(logits, dim=1).argmax(dim=1)
        #Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        #Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=["Pneumonia","Normal"])
        disp.plot().figure_.savefig(os.path.join(SAVE_VIS_PATH,f"confusion_matrx.png"))
        plt.show()


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

if __name__ == "__main__":
    #Retrieve data in x (arrays of images) and y (class labels)
    x_train, x_test, y_train, y_test = load_train_data()

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


    print(x_test.shape,"this is the length of y_test")
    print(len(y_test),"this is the length of y_test")

    #Batch Size hyperparameter
    batch_size = 16

    train_tensors = torch.utils.data.TensorDataset(x_train,y_train)
    test_tensors = torch.utils.data.TensorDataset(x_test,y_test)

    # Turn dataset into iterables of mini batches
    #Shuffle is true for training to prevent learning of spurious correlation or noise
    train_dataloader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_tensors, batch_size=batch_size, shuffle=True)

    print(f"Dataloaders: {train_dataloader,test_dataloader}")
    print(f"Length of train dataloaders: {len(train_dataloader)} of batch size {batch_size}")
    print(f"Length of test dataloaders: {len(test_dataloader)} of batch size {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 10

    final_model,final_optimizer = final_model_run(train_dataloader,
                                                  test_dataloader,
                                                  NUM_EPOCHS,
                                                  device)
    
    #Save only the learnt weights and optimizer parameters and not the whole model
    final_model = torch.save(final_model.state_dict(), os.path.join(SAVE_MODEL_PATH,"best_model.pt"))
    
    #Saving the final optimizer weights for best practice especailly when Adam optimizer is used.
    final_optimizer = torch.save(final_optimizer.state_dict(), os.path.join(SAVE_MODEL_PATH,"best_optimizer.pt"))

    evaluate_test_model(x_test,y_test)
    

