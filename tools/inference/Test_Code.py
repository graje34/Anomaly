import os
import unittest
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

from lightning_inference import infer


class TestInference(unittest.TestCase):
    def setUp(self):
        self.weight_file = "tools/inference/Weight/model.ckpt"
        self.data_folder = "tools/inference/Data"
        self.threshold = 0.5
        
    def test_image_extensions(self):
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for folder in ["OK", "NOK"]:
            for img_file in os.listdir(os.path.join(self.data_folder, folder)):
                _, ext = os.path.splitext(img_file)
                self.assertTrue(ext in valid_extensions, f"==========================Image {img_file} in folder {folder} has an invalid file extension. Please provide a .png, .jpg, or .jpeg image file only.")

    def test_inference(self):
        for folder in ["OK", "NOK"]:
            for img_file in os.listdir(os.path.join(self.data_folder, folder)):
                img_path = os.path.join(self.data_folder, folder, img_file)
                confidence = infer(img_path, self.weight_file)
                pred_label = confidence[0]['pred_labels'][0].item()
                confidence=confidence[0]['pred_scores'][0].item()
                
                
                if confidence == 0.0:
                    confidence = 1
                    confidence=confidence*100
                    
                else:
                    confidence = confidence * 100
                    
                print("confidence is",confidence)    
                
                if folder == "OK":
                    self.assertFalse(pred_label, f"=============================================Image {img_file} in folder {folder} is misclassified as Anomaly with {confidence}% confidence.")
                elif folder == "NOK":
                    self.assertTrue(pred_label, f"==============================================Image {img_file} in folder {folder} is misclassified as Normal with {confidence}% confidence.")
                    


if __name__ == '__main__':
    unittest.main()

