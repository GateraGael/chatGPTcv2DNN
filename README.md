# Description

This is an attempt at combining OpenCV's DNN module with the OpenAI GPT-3.5-turbo API to classify dog breeds in images and ask for more information about the predicted breed. The script has several functions to load the model, preprocess the image, detect dogs, and load synset words (labels). Here's a breakdown of the code:

Import necessary libraries: sys, os, cv2, openai, numpy, and matplotlib.

Set the OpenAI API key by reading it from a file named key.txt.

Define a message_history list that stores the conversation history with GPT-3.5-turbo.

Implement the predict() function, which takes an input string and appends it to the message_history. It then calls the OpenAI GPT API to generate a response based on the conversation history. Finally, it returns the response.

Implement the load_model(), preprocess_image(), detect_dogs(), and load_synset_words() functions, which are responsible for loading the DNN model, preprocessing the input image, detecting the dog breed in the image, and loading the synset words (labels) from a text file, respectively.

The main() function:

a. Loads the DNN model and its configuration.

b. Loads the synset words from a text file.

c. Reads the input image and preprocesses it.

d. Detects the dog breed in the image using the DNN model.

e. Prints the predicted class ID and class name.

f. Calls the predict() function to ask GPT-3.5-turbo for more information about the predicted dog breed and relevant deep learning models for image classification.

g. Displays the predicted class name on the image.

h. Saves the image with the predicted class name to an output directory.

When you run this script, it will read an input image, classify the dog breed using OpenCV's DNN module and a pre-trained ResNet-50 model, display the predicted class name on the image, and save the image to the output directory. Additionally, it will ask GPT-3.5-turbo for more information about the predicted dog breed and related deep learning models for image classification.

### Set Up Instructions

Install Requirements txt file
```
pip install -r requirements.txt
```

Download required Model and Config file
```
wget https://www.deepdetect.com/downloads/platform/pretrained/resnet_50/ResNet-50-model.caffemodel
wget https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/prototxt/ResNet-50-deploy.prototxt
```