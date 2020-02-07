import sys
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np

import argparse

''' The Dataset class holds the training, validation and testing datasets
'''
class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.validation_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        try:                                                    
            self.train_data = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        except OSError as err:
            print('Cannot open the training folder: ', self.train_dir) 
            sys.exit(1)
        
        test_validation_transforms = transforms.Compose([transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                             [0.229, 0.224, 0.225])])
        try:                                                    
            self.validation_data = datasets.ImageFolder(self.validation_dir, transform=test_validation_transforms)
        except OSError as err:
            print('Cannot open the validation folder: ', self.validation_dir) 
            sys.exit(1)

        try:                                                    
            self.test_data = datasets.ImageFolder(self.test_dir, transform=test_validation_transforms)
        except OSError as err:
            print('Cannot open the test folder: ', self.test_dir) 
            sys.exit(1)
        
    def get_class_to_idx(self):
        return self.train_data.class_to_idx
    
    def get_trainloader(self, batch_size=256):
        trainloader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True)
        
        return trainloader
        
    def get_validationloader(self, batch_size=256):
        validationloader = torch.utils.data.DataLoader(self.validation_data, batch_size, shuffle=True)
        
        return validationloader

    def get_testloader(self, batch_size=256):
        testloader = torch.utils.data.DataLoader(self.test_data, batch_size, shuffle=True)
        
        return testloader

def build_parser():
    ''' Construct a parser object for the arguments for train.py
        OUTPUT
            The parser object
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="the data directory for the images")
    parser.add_argument("--save_dir", help="the directory to save the checkpoint in", default='.')
    parser.add_argument("--arch", help="the architecture to use for the model",
                        choices=['densenet121', 'resnet18'], default='densenet121')
    parser.add_argument("--learning-rate", help="the learning rate for training", type=float, default=0.003)
    parser.add_argument("--epochs", help="the number of epochs to train over", type=int, default=8)
    parser.add_argument("--hidden_units", help="the number of hidden units in the fc layer", type=int, default=256)
    parser.add_argument("--gpu", help="use gpu for training the model", action='store_true')
    
    return parser

def build_model(arch, hidden_units):
    ''' Build the training model 
        INPUTS:
          arch: the architecture to use for the model
                the options are 'resnet18' and 'densenet121'
          hidden_units: the number of hidden units in the
                        fully connected classification layer
        OUTPUT:
            model
    '''        
    num_ftrs = 0
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
    else:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        
    model.arch = arch
    model.hidden_units = hidden_units
    model.in_features = num_ftrs
    
    for param in model.parameters():
        param.requires_grad = False
            
    fc = nn.Sequential(nn.Linear(num_ftrs, hidden_units),
                       nn.ReLU(),
                       nn.Dropout(p=0.2),
                       nn.Linear(hidden_units, 102),
                       nn.LogSoftmax(dim=1))
    if arch == 'resnet18':
        model.fc = fc
    else:
        model.classifier = fc
    
    return model

def train_model(model, dataset, learning_rate, epochs, gpu):
    ''' Train the model over the training set and measure accuracy over
        the validation set
        INPUTS:
          model: The model to train
          dataset: the dataset object containing the training and validation data
          learning_rate: the learning rate for the optimizer
          epochs: the number of epochs to train the model over
          gpu: flag to decide whether to train using a GPU
       OUTPUT:
          the trained model
    '''      
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    
    criterion = nn.NLLLoss()
    
    # Only train the classifier parameters, feature parameters are frozen
    if model.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        
    # Store the class to index in the model
    model.class_to_idx = dataset.get_class_to_idx()
    
    for e in range(epochs):
        running_loss = 0
        for images, labels in dataset.get_trainloader():
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Loss over training set after epoch {e+1}: {running_loss:.3f}')
        # Validation set
        model.eval()
        with torch.no_grad():
            running_loss = 0
            right = 0
            total = 0
            accuracy = 0
            # measure accuracy and loss over validation data
            for images, labels in dataset.get_validationloader():
                images, labels = images.to(device), labels.to(device)
                
                logps = model.forward(images)
                loss = criterion(logps, labels)
                running_loss += loss.item()
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                right += equals.int().sum()
                total += len(equals)
                accuracy = float(right)/total            
            print(f'Loss over validation set after epcoh {e+1}: {running_loss:.3f}')
            print(f'Accuracy on validation set after epoch {e+1}: {accuracy:.3f}')
        model.train()
        
    return model

def checkpoint_model(model, save_dir):
    ''' Save the model
        INPUTS
          model: the model to save
          save_dir: the directory to save the model in
                    the checkpoint will have a name based in the architecture 
                    of the model e.g. densenet121_checkpoint.pth
    '''                
    checkpoint = {'arch': model.arch,
                  'classifier_input': model.in_features,
                  'classifier_hidden': model.hidden_units,
                  'classifier_output': 102,
                  'classifier_dropout': 0.2,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }

    torch.save(checkpoint, save_dir+'/'+model.arch+'_checkpoint.pth')
    
def main():
    ''' The main function for train.py
    '''
    # Build the parser object
    parser = build_parser()
    args = parser.parse_args()
    
    # Create a dataset object
    dataset = Dataset(args.data_dir)
    
    # Build the model
    model = build_model(args.arch, args.hidden_units)
    
    # Train the model
    train_model(model, dataset, args.learning_rate, args.epochs, args.gpu)
    
    # Checkpoint model
    checkpoint_model(model, args.save_dir)
    
if __name__=="__main__":
    main()