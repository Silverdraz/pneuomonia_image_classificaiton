"""
A frontend streamlit web app as a poc. Here, the frontend components are deisgned and allows for processing of new images using the trained model
saved under the directory of saved_model. The classificatio label is then shown on the streamlit web app
"""


#Import statements
import streamlit as st #frontend
import torch #pytorch 
from PIL import Image #Image manipulation

#Import modules
import model_builder #model architecture
import inference # For prediction on a new test image

#Path to best model
LOAD_MODEL_PATH = r"..\saved_model\best_model.pt"

if __name__ == "__main__":

    #Set title
    st.title("Pneumonia Classification")

    #Set header
    st.header("Please upload a chest x-ray image")

    #Upload file with image file extensions
    file = st.file_uploader('', type=['jpg','jpeg','png'])

    # select gpu when available, else work with cpu resources
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load classifier (not fine-tuned weights)
    inference_model = model_builder.VGGNet3BlockBN()
    #Load the fine-tuned weights on chest-xray dataset
    inference_model.load_state_dict(torch.load(LOAD_MODEL_PATH))
    #Set the mode to eval for inference (especially for dropout and batchnorm2d)
    inference_model.eval()

    # load class names from txt file
    with open(r'..\saved_model\labels.txt', 'r') as f:
        class_names = [a[:].split(' ')[1] for a in f.readlines()]
        f.close()

    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = inference.classify(image, inference_model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### score: {}%".format(conf_score))