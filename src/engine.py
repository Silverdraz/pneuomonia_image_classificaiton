"""
Trains the model using the provided optimizer, loss functions in a range of epochs
"""

#Import statements
from tqdm.auto import tqdm # Import tqdm for progress bar
from sklearn.metrics import f1_score #f1 metric

#Torch import statements
import torch #General torch library

def accuracy_fn(y_true,y_pred):
    """Creates accuracy function for reusability.

    Takes in predictions and actual labels. Calculate the accuracy for these inputs and
    return the accuracy as a number

    Args:
        y_true: list of ground-truth labels.
        y_pred: list of predicted labels.

    Returns:
        Accuracy as a score
    """
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def f1_fn(y_true,y_pred):
    """Creates f1 scoring metric for reusability

    Takes in predictions and actual labels. Calculate the f1_score for these inputs and
    return the f1_score as a number
    
    Args:
        y_true: list of ground-truth labels.
        y_pred: list of predicted labels.

    Returns:
        f1_score as a score
    """
    f1 = f1_score(y_true.cpu(),y_pred.cpu())
    return f1

def train_step(model,
               dataloader,
               loss_fn ,
               optimizer,
               device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model (defined in model_builder.py) to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics and f1 scoring metric.
    In the form (train_loss, train_accuracy, train_f1). 
    """
    
    #Training
    train_loss, train_acc, train_f1 = 0, 0, 0
    #Add a loop to loop through the training batches
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)
        
        #Allow configs for learning (e.g. weight updates)
        model.train()

        #Forward pass
        y_pred = model(X)
        y_pred_labels = y_pred.argmax(dim=1)
        
        #Calculate the training loss per batch
        loss = loss_fn(y_pred, y)
        train_loss += loss
        #Calculate the training accuracy per batch
        train_acc += accuracy_fn(y_true=y,
                            y_pred=y_pred_labels)
        
        #Calculate the training f1 per batch
        train_f1 += f1_fn(y_true=y,
                            y_pred=y_pred_labels)
        
        #Optimizer zero grad
        optimizer.zero_grad()
        
        #Loss backward
        loss.backward()
        
        #Optimizer step
        optimizer.step()

    #Average train loss per batch in an epoch and Average train accuracy per batch in an epoch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_f1 /= len(dataloader)

    train_loss = train_loss.cpu().detach().numpy()
    return train_loss, train_acc, train_f1


def test_step(model, 
              dataloader, 
              loss_fn,
              device):
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics and testing f1 scoring metric.
        In the form (test_loss, test_accuracy, test_f1). 
    """
    #Testing
    test_loss, test_acc, test_f1 = 0, 0, 0
    #Set to eval mode to prevent auto diff.
    model.eval()
    with torch.inference_mode():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            #Forward Pass
            test_pred = model(X)
            y_pred_labels = test_pred.argmax(dim=1)
            
            #Calculate the validation loss per batch
            loss = loss_fn(test_pred,y)
            test_loss += loss
            #Calculate the validation accuracy per batch
            test_acc += accuracy_fn(y_true=y,
                                y_pred=y_pred_labels)
            
            #Calculate the validation f1 per batch
            test_f1 += f1_fn(y_true=y,
                                y_pred=y_pred_labels)
            
        #Average test loss per batch in an epoch and Average test accuracy per batch in an epoch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        test_f1 /= len(dataloader)
    
    test_loss = test_loss.cpu().detach().numpy()
    return test_loss, test_acc, test_f1


def train(model, 
          train_dataloader, 
          test_dataloader, 
          optimizer,
          loss_fn,
          epochs,
          device):
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics as well as training and testinf f1 scoring metric. 
    Each metric has a value in a list for each epoch for graph plotting
    """
    # Create empty results dictionary
    results = {"train_loss_list": [],
               "train_accuracy_list": [],
               "train_f1_list": [],
               "test_loss_list": [],
               "test_accuracy_list": [],
               "test_f1_list": []
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_f1 = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device
                                        )
        test_loss, test_acc, test_f1 = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device
                                        )

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_acc: {test_acc:.4f} | "
          f"train_f1: {train_f1:.4f} | "
          f"test_f1: {test_f1:.4f}"
        )

        # Update results dictionary
        results["train_loss_list"].append(train_loss)
        results["train_accuracy_list"].append(train_acc)
        results["train_f1_list"].append(train_f1)
        results["test_loss_list"].append(test_loss)
        results["test_accuracy_list"].append(test_acc)
        results["test_f1_list"].append(test_f1)

    # Return the filled results at the end of the epochs
    return results


