# Common external imports
import os
import time
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

from collections import Counter
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random

# Common project imports
from config import DATA_DIR, SAMPLE_SIZE, BATCH_SIZE, EPOCHS

# From https://github.com/visionjo/facerec-bias-bfw/blob/master/code/notebooks/1a_generate_mean_faces.ipynb
import pathlib
path_package=f'../'
import sys
if path_package not in sys.path:
    sys.path.append(path_package)

import warnings
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# to run the following, you have to run this in termal
    #git clone https://github.com/visionjo/facerec-bias-bfw.git
    #cd facerec-bias-bfw/code
    #pip install .

from facebias.image import read, write, resize 
