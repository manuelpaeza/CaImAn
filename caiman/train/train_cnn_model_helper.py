import numpy as np
import os
import keras 
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense 
from keras.models import save_model, load_model 
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight as cw
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split 

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.utils.image_preprocessing_keras import ImageDataGenerator

os.environ["KERAS_BACKEND"] = "torch"

class cnn_model_pytorch(torch.nn.Module): 
    def __init__(self, in_channels, num_classes):
        super(cnn_model_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1, 1))
        self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1, 1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1, 1))
        self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=6400, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout(self.maxpool2d1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.dropout(self.maxpool2d2(x), p=0.25)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=0.5)
        x = F.softmax(self.dense2(x), dim=1)
        return x 

def save_model_pytorch(model, name: str):
    model_name = os.path.join(caiman_datadir(), 'model', name)
    model_path = model_name + ".pth"
    torch.save(model, model_path)
    print('Saved trained model at %s ' % model_path)
    return model_path 

def load_model_pytorch(model_path: str):
    load_model = torch.load(model_path)
    print('Load trained model at %s ' % model_path)
    return load_model  

def train_test_split(dataset: Dataset, test_fraction: float):
    train_ratio = 1 - test_fraction
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    lengths = [train_size, test_size]
    train_dataset, test_dataset = random_split(dataset, lengths)
    return train_dataset, test_dataset

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train(model, train_loader, loss_function, optimizer, train_N, augment):
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)  
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def validate(model, valid_loader, loss_function, optimizer, valid_N, augment):
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def cnn_model_keras(input_shape, num_classes): 
    sequential_model = keras.Sequential([   
        Input(shape=input_shape, dtype="float32"), 
        Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"), 
        Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"), 
        MaxPooling2D(pool_size=(2, 2)), 
        Dropout(rate=0.25),
        Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), 
        padding="same", activation="relu"),
        Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), 
        activation="relu"),  
        MaxPooling2D(pool_size=(2, 2)), 
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=512, activation="relu"),
        Dropout(rate=0.5),
        Dense(units=num_classes, activation="relu"),
    ])
    return sequential_model 

def save_model_keras(model, name: str):
    model_name = os.path.join(caiman_datadir(), 'model', name)
    model_path = model_name + ".keras"
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    return model_path 

def load_model_keras(model_path: str):
    loaded_model = load_model(model_path)
    print('Load trained model at %s ' % model_path)
    return loaded_model  
