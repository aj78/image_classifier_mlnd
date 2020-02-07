import sys
import torch
from torch import nn
#from torch import optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

import numpy as np
import json

import argparse

def build_parser():
    ''' Construct a parser object for the arguments for predict.py
        OUTPUT
            The parser object
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="the path to the image to be predicted")
    parser.add_argument("checkpoint", help="the checkpoint to load the model from")
    parser.add_argument("--top_k", help="the top k classes predicted by the model", type=int, default=1)
    parser.add_argument("--category_names", help="the json file with the category names")
    parser.add_argument("--gpu", help="use gpu for inference", action='store_true')
    
    return parser

def load_checkpoint(filepath):
    ''' Load into a model the training parameters from the checkpoint file
        INPUT
          filepath: the path to the checkpoint file
        OUTPUT
          the model after loading the checkpoint
    '''
    # Load checkpoint
    try:
        checkpoint = torch.load(filepath)
    except OSError as err:
        print('Cannot open the checkpoint file: ', filepath) 
        sys.exit(1)
    
    # start with the appropriate model
        
    classifier = nn.Sequential(nn.Linear(checkpoint['classifier_input'], checkpoint['classifier_hidden']),
                               nn.ReLU(),
                               nn.Dropout(p=checkpoint['classifier_dropout']),
                               nn.Linear(checkpoint['classifier_hidden'], checkpoint['classifier_output']),
                               nn.LogSoftmax(dim=1))
    if checkpoint['arch'] == 'resnet18':
        model = models.resnet18()
        model.fc = classifier
    else:
        model = models.densenet121()
        model.classifier = classifier
        
    model.arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model
        INPUT
          image: the file path for an image
        OUTPUT
          returns a torch tensor corresponding to the image
    '''
    
    # Process a PIL image for use in a PyTorch model
    try:
        im = Image.open(image)
    except OSError as err:
        print('Cannot open the image: ', image) 
        sys.exit(1)

    im.thumbnail((256,256))
    # center crop to 224x224
    w, h = im.size
    im = im.crop(((w - 224)//2, (h - 224)//2, (w + 224)//2, (h + 224)//2))
    np_image = np.array(im)
    #Normalize
    d = np.array([255.0, 255.0, 255.0])
    np_image = np_image / d
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = ( np_image - mean ) / std
    #Transpose
    np_image = np_image.transpose((2, 1, 0))

    # return a torch tensor
    return torch.from_numpy(np_image)

def predict(image_path, model, gpu, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        INPUTS
          image_path: the path to the image to predict
          model: the model to use for prediction
          gpu: whether to use GPU for prediction
          topk: the top k predictions to return
        OUTPUT  
          Returns a list of top probabilities and the top classes
    '''
    
    # Implement the code to predict the class from an image file
    device = torch.device("cuda" if gpu else "cpu")
    
    image = process_image(image_path)
    image = image.view(1, 3, 224, 224)
    image = image.to(device, dtype=torch.float)
    model.to(device)
    model.eval()    
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_p, top_idx = ps.topk(topk, dim=1)
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_class = [idx_to_class[key] for key in top_idx.data.tolist()[0]]
    
    # return lists from the tensors
    return top_p.data.tolist()[0], top_class

def print_result(top_p, top_class, category_names):
    ''' print the formatted top probabilities and classes
        If category_names are not provided, print the classes
    '''
    top_p_formatted = [ '%.3f' % elem for elem in top_p ]

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[key] for key in top_class]
    print('\n')
    print('prob  - class')
    print('-------------')
    for prob, cat in zip(top_p_formatted, top_class):
        print(f'{prob} - {cat}')

        
def main():
    ''' The main function for predict.py
    '''
    # Build the parser object
    parser = build_parser()
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)

    top_p, top_class = predict(args.image_path, model, args.gpu, args.top_k)
    print_result(top_p, top_class, args.category_names)

if __name__=="__main__":
    main()