"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks





import sys
import cv2
#import neoapi
import time
import cv2
import numpy as np
from lightning_inference import *

import sys
import cv2

weight_file="/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib_main/tools/inference/Weight/model.ckpt"
img = '/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib_main/tools/inference/Data/OK/'
threshold= 0.75


time.sleep(0.2)

confidence=infer(img,weight_file)
pred_score = confidence[0]['pred_scores'][0].item()
pred_score*100
print(pred_score)

if pred_score < threshold:
    print("Image is Anamoly")
else:
    print("image is OK")    



print("Confidence is ", confidence)


pred_label= confidence[0]['pred_labels'][0]
print("pred_label",pred_label)

