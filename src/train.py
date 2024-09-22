#import statements
import os #os 
import numpy as np #array/matrix manipulation
from sklearn.model_selection import train_test_split # Split data into train and test sets
import matplotlib.pyplot as plt #data viz library

#Torch import statements
import torch #General torch library
from torch.utils.data import DataLoader #Mini batch data loader
from torch import nn #Basic Building Blocks for pytorch
from torchvision import transforms #Data Augmentation

#import modules
import loading_preprocessing_data #data preprocessing and loading module
import model_builder #model architectures
import engine #training of model
import dataset #custom dataset module


# Set Data Path to data directory
DATA_PATH = r"..\data\raw"
SAVE_VIS_PATH = r"..\visualisations"

def plot_loss_curve(train_loss_list,val_loss_list,model_string):
    #Plot the loss curves for training loss and validation loss
    num_epochs = len(train_loss_list)
    plt.plot(np.arange(1,num_epochs+1),train_loss_list,label="training loss")
    plt.plot(np.arange(1,num_epochs+1),val_loss_list,label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Training/Validation Loss across Epochs")
    plt.legend()
    plt.savefig(os.path.join(SAVE_VIS_PATH,f"{model_string}_loss.png"))
    plt.close()
    #plt.show()

def plot_accuracy_curve(train_accuracy_list,val_accuracy_list,model_string):
    #Plot the accuracy curves for training accuracy and validation accuracy
    num_epochs = len(train_accuracy_list)
    plt.plot(np.arange(1,num_epochs+1),train_accuracy_list,label="training accuracy")
    plt.plot(np.arange(1,num_epochs+1),val_accuracy_list,label="validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Training/Validation accuracy across Epochs")
    plt.legend()
    plt.savefig(os.path.join(SAVE_VIS_PATH,f"{model_string}_f1.png"))
    plt.close()
    #plt.show()


if __name__ == "__main__":
    train = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"train"))
    test = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"test"))
    val = loading_preprocessing_data.get_training_data(os.path.join(DATA_PATH,"val"))

    x_train, x_test, y_train, y_test = [], [], [], []
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(train,"train",x_train,x_test,y_train,y_test)
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(val,"val",x_train,x_test,y_train,y_test)
    x_train, x_test, y_train, y_test = loading_preprocessing_data.load_to_list(test,"test",x_train,x_test,y_train,y_test)

    x_train, x_test = loading_preprocessing_data.array_normalize_data(x_train,x_test)

    # resize data for deep learning 
    img_size = 224
    x_train = x_train.reshape(-1,1,img_size,img_size)
    y_train = np.array(y_train)

    x_test = x_test.reshape(-1,1,img_size, img_size)
    y_test = np.array(y_test)

    print(x_train[0], x_train[0].shape)
    print(len(y_train))
    print(len(x_train))
    print(x_train.shape,"shape")

    #Convert to tensors in particular long tensor for y_train since nn.crossentropy requires y labels to
    #be long type
    x_train = torch.from_numpy(x_train).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    x_test = torch.from_numpy(x_test).type(torch.float)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)

    #Split the dataset into train and validation
    #Set the random seed for reproduciblity
    RANDOM_SEED = 42
    #Split into x_train and y_train using stratify since the dataset is imbalanced
    x_train, x_val, y_train, y_val = train_test_split(x_train, 
                                                        y_train, 
                                                        test_size=0.15, # 15% test, 85% train,
                                                        stratify=y_train,
                                                        random_state=RANDOM_SEED) # make the random split reproducible
    print(len(x_train), len(x_val), len(y_train), len(y_val))

    #Batch Size hyperparameter
    batch_size = 32

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

    # Instantiate an instance of the model from the "model_builder.py" script
    torch.manual_seed(42)
    model = model_builder.VGGNetBaseline().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.01)
    

    print(device,"this is device")

    # Start training with help from engine.py
    results = engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=10,
                device=device)
    
    train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    plot_loss_curve(train_loss_list,val_loss_list,"vggnet")
    plot_accuracy_curve(train_accuracy_list, val_accuracy_list,"vggnet")

    ##############################

    
    # Instantiate an instance of the model from the "model_builder.py" script
    torch.manual_seed(42)
    model_1 = model_builder.VGGNet3Block().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_1.parameters(),lr=0.01)

    print(device,"this is device")

    # Start training with help from engine.py
    results = engine.train(model=model_1,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=10,
                device=device)
    
    train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    plot_loss_curve(train_loss_list,val_loss_list,"vggnet3block")
    plot_accuracy_curve(train_accuracy_list, val_accuracy_list,"vggnet3block")

    #########################
    
    # Instantiate an instance of the model from the "model_builder.py" script
    torch.manual_seed(42)
    model_2 = model_builder.VGGNet3BlockBN().to(device)
    #Set up Loss function and optimizer for multi class classification
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_2.parameters(),lr=0.01)

    print(device,"this is device")

    # Start training with help from engine.py
    results = engine.train(model=model_2,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=10,
                device=device)
    
    train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    plot_loss_curve(train_loss_list,val_loss_list,"vggnet3block")
    plot_accuracy_curve(train_accuracy_list, val_accuracy_list,"vggnet3blockbn")



    ########################
    # model_1 = model_builder.VGGNetBN().to(device)
    # #Set up Loss function and optimizer for multi class classification
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model_1.parameters(),lr=0.01)

    # print(device,"this is device")

    # # Start training with help from engine.py
    # results = engine.train(model=model_1,
    #             train_dataloader=train_dataloader,
    #             test_dataloader=val_dataloader,
    #             loss_fn=loss_fn,
    #             optimizer=optimizer,
    #             epochs=10,
    #             device=device)
    
    # train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    # train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    # plot_loss_curve(train_loss_list,val_loss_list,"vggnet_bn")
    # plot_accuracy_curve(train_accuracy_list, val_accuracy_list,"vggnet_bn")

    # ###########
    # #Batch Size hyperparameter
    # batch_size = 32

    # #train_tensors = torch.utils.data.TensorDataset(X_train,y_train)
    # #val_tensors = torch.utils.data.TensorDataset(X_val,y_val)
    # train_transform = transforms.Compose([transforms.RandomRotation(15),
    #                                     transforms.RandomAffine(degrees=10)])

    # # Turn dataset into iterables of mini batches
    # #Shuffle is true for training to prevent learning of spurious correlation or noise
    # train_aug_dataset = dataset.XRAYDatasetDataAug(train_tensors,train_transform)
    # train_dataloader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True)

    # val_aug_dataset = dataset.XRAYDatasetDataAug(val_tensors,None)
    # val_dataloader = DataLoader(val_aug_dataset, batch_size=batch_size, shuffle=False)

    # print(f"Dataloaders: {train_dataloader,val_dataloader}")
    # print(f"Length of train dataloaders: {len(train_dataloader)} of batch size {batch_size}")
    # print(f"Length of validation dataloaders: {len(val_dataloader)} of batch size {batch_size}")
    
    # model_2 = model_builder.VGGNetBaseline().to(device)
    # #Set up Loss function and optimizer for multi class classification
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(params=model_2.parameters(),lr=0.01)

    # print(device,"this is device")

    # # Start training with help from engine.py
    # results = engine.train(model=model_2,
    #             train_dataloader=train_dataloader,
    #             test_dataloader=val_dataloader,
    #             loss_fn=loss_fn,
    #             optimizer=optimizer,
    #             epochs=10,
    #             device=device)
    
    # train_loss_list, val_loss_list = results["train_loss_list"], results["test_loss_list"]
    # train_accuracy_list, val_accuracy_list = results["train_accuracy_list"], results["test_accuracy_list"]
    # plot_loss_curve(train_loss_list,val_loss_list,"vggnet_baseline_dataaug")
    # plot_accuracy_curve(train_accuracy_list, val_accuracy_list,"vggnet_baseline_dataaug")