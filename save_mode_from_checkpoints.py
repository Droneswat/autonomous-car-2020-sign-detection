from lib.core.model.facebox.net import FaceBoxes

import tensorflow as tf
import cv2
import numpy as np
import os
from train_config import config as cfg 

model=FaceBoxes()

if cfg.MODEL.pretrained_model is not None:
    model.load_weights(cfg.MODEL.pretrained_model)

current_model_saved_name=os.path.join(cfg.MODEL.model_path, 'save_from_checkpoint')
if not os.access(cfg.MODEL.model_path,os.F_OK):
    os.mkdir(cfg.MODEL.model_path)
tf.saved_model.save(model,current_model_saved_name)