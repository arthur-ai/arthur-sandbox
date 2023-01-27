import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import pathlib
import random


from PIL import Image
from torch import from_numpy

# net = None

# class 

# have to define class to load model
class MedNet(nn.Module):
    def __init__(self):
        super(MedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 96 -> 92; 32 -> 28
        self.pool1 = nn.MaxPool2d(2, 2) # 92 -> 46; 28 -> 14
        self.conv2 = nn.Conv2d(6, 16, 5) # 46 -> 42; 14 -> 10
        self.pool2 = nn.MaxPool2d(2, 2) # 42 -> 21; 10 -> 5
        self.conv3 = nn.Conv2d(16, 32, 5) # 21 -> 17; n / a
        self.pool3 = nn.MaxPool2d(2, 2) # 17 -> 8
        self.fc1 = nn.Linear(32*8*8, 1024) # Avoids the center2 'bump'; basically, just accomodates what the network needs
        self.fc2 = nn.Linear(1024, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.sigmoid(x)

# load model
net = MedNet()
path = pathlib.Path(__file__).parent.absolute()
net.load_state_dict(torch.load(f'{path}/portable_cancer_classifier'))

def quantize(np_array):
    return np_array + (np.random.random(np_array.shape) / 256)

def load_image(image_path):
    """Takes in image path, and returns image in format predict expects
    """
    return quantize(np.array(Image.open(image_path).convert('RGB')) / 256)


def predict(images_in):
    """Takes in numpy array of images, output from load_image
    """
    batch_size, pixdim1, pixdim2, channels = images_in.shape
    raw_tensor = torch.from_numpy(images_in)
    processed_images = torch.reshape(raw_tensor, (batch_size, channels, pixdim1, pixdim2)).float()
    net.eval()
    with torch.no_grad():
        return net(processed_images).numpy()