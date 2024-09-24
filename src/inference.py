"""
Perform inference on the single image that is uploaded to the streamlit. Similar preprocessing that was applied during training
will have to be applied now as well so as to maintain consistency.
"""

#Import statements
from PIL import ImageOps, Image #Image manipulation and opening
import numpy as np #array/numpy manipulation
import torch #pytorch


def forward_pass(normalized_image_tensor,model,class_names):
    """
    This function takes a normalized image tensor for prediction and classification.
    It is a forward pass.

    Parameters:
        normalized_image_tensor: An image to be classified.
        model: A trained machine learning model for image classification.
        class_names: A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """

    # Run forward pass and run forward
    with torch.no_grad():
        logits = model(normalized_image_tensor)
        y_softmax = torch.softmax(logits, dim=1)
        index = torch.argmax(y_softmax,dim=1)
        #index = 0 if y_softmax[0][0] > 0.95 else 1
        class_name = class_names[index]
        confidence_score = y_softmax[0][index].item()

    return class_name, confidence_score



def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image: An image to be classified.
        model: A trained machine learning model for image classification.
        class_names: A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    # applying grayscale method 
    grayscale_image = ImageOps.grayscale(image) 

    # convert grayscale image to numpy array
    image_array = np.asarray(grayscale_image)

    # normalize image
    normalized_image_array = image_array / 255.0

    # set model input and reshape it to [B,C,W,H]
    normalized_image_array = normalized_image_array.reshape(1,1,224,224)
    #Set it to tensor
    normalized_image_tensor = torch.from_numpy(normalized_image_array).type(torch.float)

    #Run a forward pass
    class_name, confidence_score = forward_pass(normalized_image_tensor,model,class_names)

    return class_name, confidence_score 

