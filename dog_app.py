'''
DESCRIPTION

This module is able to process an image given to it and detect either a dog or human
It further is able to classify the detected dog breed using a trained model with 82% accuracy


INPUTS

img_path -   path to an image


OUTPUTS

Returns a dog breed description if a dog is detected
If a human is detected it outputs the closest resemblace to a dog



SCRIPT EXECUTION SAMPLE

python dog_app.py <img_path>

'''

import torch
from torchvision import models, transforms
from torch import nn
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')


def face_detector(img_path):
    '''
    DESCRIPTION
    Detects face(s) within a given image 
    
    INPUTS
    img_path -   path to an image
    
    OUTPUTS
    Boolean - returns True if face detected else False
    
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# define RESNET50 model
RESNET50 = models.resnet50(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    RESNET50 = RESNET50.cuda()



def RESNET50_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    image = Image.open(img_path)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    
    image_tensor = transform(image)
    
    image_tensor.unsqueeze_(0)
    
    RESNET50.eval()
    
    if use_cuda:
        image_tensor = image_tensor.cuda()
        
    output = RESNET50(image_tensor)
    pred_value, pred_idx = torch.max(output,1)
    pred_out = pred_idx.item()
    
    
    return pred_out # predicted class index



def dog_detector_resnet(img_path):
    
    '''
    DESCRIPTION
    Detects dog(s) within a given image using a pretrained Resnet model
    
    INPUTS
    img_path -   path to an image
    
    OUTPUTS
    Boolean - returns True if dog detected else False
    
    '''
    
    index = RESNET50_predict(img_path)
    
    return index >= 151 and index <=268 # true/false



def define_model():
    
    '''
    DESCRIPTION
    Defines the framework for the model to be used for classifcation
    
    OUTPUT
    model_transfer - the finetuned model architecture
    
    
    '''
    model_transfer = models.densenet121(pretrained=True)

    for params in model_transfer.features.parameters():
        params.requires_grad = False


    num_inputs = model_transfer.classifier.in_features
    final_layer = nn.Linear(num_inputs,133)
    model_transfer.classifier = final_layer
    
    return model_transfer
    



def predict_breed_transfer(img_path, model_transfer):
    
    '''
    DESCRIPTION
    Uses a model passed to it to predict a dog breed of a given image
    
    INPUTS
    img_path -   path to an image
    
    OUTPUT
    pred_out_name - the dog breed identified by the model
    
    '''
    f = open('dog_breed_list.txt')
    class_names = f.readlines()
    class_names = [name[:-1] for name in class_names]
    
    # load the image and return the predicted breed
    image_t = Image.open(img_path)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

    
    image_t_tensor = transform(image_t)
    
    image_t_tensor.unsqueeze_(0)
    
    model_transfer.eval()
    
    if use_cuda:
        image_t_tensor = image_t_tensor.cuda()
    
    
    output_t = model_transfer(image_t_tensor)
    pred_value_t, pred_idx_t = torch.max(output_t,1)
    pred_out_t = pred_idx_t.item()
    pred_out_name = class_names[pred_out_t]
    
    return pred_out_name



def run_app(img_path):
    
    '''
    DESCRIPTION
    Uses the previously defined functions to detect either a dog or human
    if a dog is  detected the breed is identified
    if a human is detected the closest dog breed resemblace is found
    
    '''
    
    ## handle cases for a human face, dog, and neither    
    
    if dog_detector_resnet(img_path) == True:
        model_transfer = define_model()
        model_transfer.load_state_dict(torch.load('saved_models/model_transfer.pt',map_location='cpu'))
        dog_breed = predict_breed_transfer(img_path, model_transfer)
        print('Dog detected....It is a {0}'.format(dog_breed))
    elif face_detector(img_path) == True:
        dog_breed = predict_breed_transfer(img_path, model_transfer)
        print('Human detected....you look like a {0}'.format(dog_breed))
    else:
        print('No human or dog detected. Please provide an image with either a dog or human')



def main():
    
    img_path = sys.argv[1]
    run_app(img_path)
    
    


if __name__ == '__main__':
    main()