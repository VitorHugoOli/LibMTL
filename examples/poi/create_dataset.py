"""
In this file we need to create the fucntion to parse the features to DataLoaders. The final format should be like:
{
'poi': {
    'train': DataLoader,
    'val': DataLoader,
    'test': DataLoader
    },
'pcg': {
    'train': DataLoader,
    'val': DataLoader,
    'test': DataLoader
    },
}
"""
import os
import torch
import numpy as np

def creating_dataloaders():
    pass