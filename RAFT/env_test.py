# adding 'RAFT/core' to the Python search path
sys.path.append('RAFT/core')

# necessary imports

import os
import sys
import numpy as np
import cv2
import pandas as pd

import torch                        # for all things PyTorch
import torch.nn as nn               # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F     # for the activation function

from collections import OrderedDict
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


# -------------------------------------------
# defining hepler functions

# convert to torch and get correct dimensions
def process_img(img, device):
    print("[process_img] entering")
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path, args):
    print("[load_model] entering")

    model = RAFT(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[load_model] current device is: " + str(device))

    try:
        pretrained_weights = torch.load(weights_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"[load_model] Fehler beim laden der Gewichte: {e}")

    print("[load_model] device_count(): " + str(torch.cuda.device_count()))

    if torch.cuda.device_count() >= 1:
       model = torch.nn.DataParallel(model)
    
    try:
        model.load_state_dict(pretrained_weights)
    except Exception as e:
        raise RuntimeError(f"[load_model] Fehler beim setzen der Gewichte: {e}")
    
    model.to(device)

    return model

# perform inference with every model
def inference(model, frame1, frame2, device, pad_mode='sintel', iters=12, flow_init=None, upsample=True, test_mode=True):
    print("[inference] entering")

    # entering evel mode: specific operstions like batch-norm. and dropout are deactivated
    model.eval()
    
    # do not calc or store gradients: increase performance
    with torch.no_grad():
        # preprocess
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        # important because raft requires every image to be divisible by 8
        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        print("[inference] Upsampled = " + str(upsample))

        # predict flow in two different modes
        if test_mode:
            # returns the initial flow (1/8 res) + upsampled flow (upsampled res)
            flow_low, flow_up = model(frame1, frame2, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
            
            return flow_low, flow_up

        else:
            # we get all flow it. for the specified amount of iterations
            flow_iters = model(frame1, frame2, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
            
            return flow_iters

def get_viz(flo):
    print("[get_viz] entering")
    flo = flo[0].permute(1,2,0).cpu().numpy()
    return flow_viz.flow_to_image(flo)

def print_model_info(model):
    print("[print_model_info] Model architecture:")
    print(model)

    print("\n [print_model_info] Model parameters and their shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

def inspect_model(model):
    print("[inspect_model] entering")

    # print("Model Architecture:\n")
    # print(model)
    
    # Gesamtanzahl der Parameter berechnen
    total_params = sum(p.numel() for p in model.parameters())
    print( "[inspect_model] Total number of parameters: " + str(total_params) )

    frame1 = cv2.imread("/home/max/Dokumente/CV_projects/RAFT/custom_demo_frames/m_baseFrameGray.jpg")
    frame2 = cv2.imread("/home/max/Dokumente/CV_projects/RAFT/custom_demo_frames/m_nextFrameGray.jpg")

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # entering evel mode: specific operstions like batch-norm. and dropout are deactivated
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[inspect_model] current device is: " + str(device))

    # do not calc or store gradients: increase performance
    with torch.no_grad():
        # preprocess
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        # important because raft requires every image to be divisible by 8
        padder = InputPadder(frame1.shape, mode='sintel')

        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow: returns the initial flow (1/8 res) + upsampled flow (upsampled res)
        flow_low, flow_up = model(frame1, frame2, iters=12, flow_init=None, upsample=True, test_mode=True)

    
    # print("[inspect_model] Output of flow_low: ")
    # print(flow_low)

    # print("[inspect_model] Output of flow_up: ")
    # print(flow_up)

    # Modellzusammenfassung manuell erstellen
    print("[inspect_model] Model Summary: ")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
    
    print( "[inspect_model] Total number of parameters: " + str(total_params) )

    return model


# sketchy class to pass to RAFT
class Args():
  def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
    self.model = model
    self.path = path
    self.small = small
    self.mixed_precision = mixed_precision
    self.alternate_corr = alternate_corr

# Sketchy hack to pretend to iterate through the class objects
  def __iter__(self):
    return self

  def __next__(self):
    raise StopIteration


# -------------------------------------------
# code starts here

print("[env_test] entering.")


# load the ".pth" model
model = load_model("RAFT/models/raft-sintel.pth", args=Args())

# print_model_info(model)
# print(model)

inspect_model(model)


print("[env_test] finished.")