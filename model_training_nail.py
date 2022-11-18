# !pipenv install torch
# !pipenv install torchvision
# !pipenv install fastai
# !pipenv install fastcore
# !pipenv install opencv-python
# !pipenv install pillow

import torch
from torchvision import transforms
from fastcore.all import *
from fastai.text.all import *
from fastai.vision.all import *
import streamlit as st
import numpy as np

# Path where the images are saved
path = Path("nail_images")

# Set up data block
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method="squish")],
).dataloaders(path, bs=32)

# Set up model and fine tune it
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)

# Save model
learn.export("nail_model_5classes.pkl")
