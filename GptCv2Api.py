import sys, os
import cv2
import openai
import numpy as np
import matplotlib.pyplot as plt


openai.api_key = open("key.txt", 'r').read().strip('\n')


message_history = [{"role": "user", "content": f"You are a joke bot. I will specify the subject matter in my messages, and you will reply with a joke that includes the subjects I mention in my messages. Reply only with jokes to further input. If you understand, say OK."},
                   {"role": "assistant", "content": f"OK"}]


def predict(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"{input}"})

    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", #10x cheaper than davinci, and better. $0.002 per 1k tokens
      messages=message_history
    )
    #Just the reply:
    reply_content = completion.choices[0].message.content#.replace('```python', '<pre>').replace('```', '</pre>')

    print(reply_content)
    message_history.append({"role": "assistant", "content": f"{reply_content}"}) 
    
    # get pairs of msg["content"] from message history, skipping the pre-prompt:              here.
    response = [(message_history[i]["content"], message_history[i+1]["content"]) for i in range(2, len(message_history)-1, 2)]  # convert to tuples of list
    return response


def load_model(model_path, config_path):
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)
    return net

def preprocess_image(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123), swapRB=True, crop=False)
    return blob, height, width

def detect_dogs(net, blob, synset_words):
    net.setInput(blob)
    predictions = net.forward()
    class_id = np.argmax(predictions)

    return class_id, synset_words[class_id]

def load_synset_words(synset_path):
    with open(synset_path, "r") as f:
        synset_words = f.readlines()
    synset_words = [word.strip() for word in synset_words]
    return synset_words


# Main function
def main():
    # Load the DNN model and its configuration
    model_path = "./model_files/ResNet-50-model.caffemodel"
    config_path = "./model_files/ResNet-50-deploy.prototxt"
    synset_path = "./model_files/synset_words.txt"
    image_path = "./images/n02086240_105.JPEG"
    output_dir = "output_images"

    net = load_model(model_path, config_path)
    synset_words = load_synset_words(synset_path)
    image = cv2.imread(image_path)
    blob, height, width = preprocess_image(image)
    class_id, class_name = detect_dogs(net, blob, synset_words)

    print(f"Predicted class ID: {class_id}")
    print(f"Predicted class name: {class_name}")
    
    print(predict(f"Can you give me more information on the following {class_name} and which Deep Learning models can classify those based on image inputs ?"))

    # Display the predicted class name on the image
    cv2.putText(image, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the image with the predicted class name to the output directory
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_prediction.jpg")
    cv2.imwrite(output_path, image)
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()