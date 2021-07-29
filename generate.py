import os
import sys
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from PIL import Image
from torch.optim import Adam
from torchvision.models import resnet

import argparse
import utils
from Network import *

base_models = {'resnet50': (resnet.resnet50, 2048),
               'resnet34': (resnet.resnet34, 512),
               'resnet18': (resnet.resnet18, 512)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


samplepath = "model_output/predictions/mymat.txt"

allmats = open("data/vMaterials/matlist.txt").read().splitlines()

'''
code to recreate the model from state dictionary
'''
def load_pm_model(model_path,args):

    checkpoint = torch.load(model_path)
    model = Network_create_model(checkpoint['args'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval() 
    return model

'''
code to generate rendering and mdl file from predictions 
'''
def generate_material(pm_model, im):

    img = np.array(np.array(im))
    image_ = np.expand_dims(img, axis=0)
    image = np.moveaxis(image_, -1, 1) 
    t_image = torch.from_numpy(image).to(device)
    outputs = pm_model(t_image.float())
    pred = {}

    pred['classid'] = outputs['classid'][0:1,...]
    for ptype in ALLTYPES:
        pred[ptype] = outputs[ptype][0:1,...]
        pred['img'] = image  

    utils.cache_pred(pred) 
    rendering_path = utils.render_prediction(samplepath)
    subprocess.call("./render.sh")
    return rendering_path


