# A tool to get and slim downloaded controlnet

import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict
import os.path as osp

in_path  = 'pretrained/controlnet/sdwebui_compatible/control_v11p_sd15_canny.pth'
out_path = 'pretrained/controlnet/control_v11p_sd15_canny_slimmed.safetensors'

sd = torch.load(in_path)

sdnew = [[ni.replace('control_model.', ''), vi] for ni, vi in sd.items()]
save_file(OrderedDict(sdnew), out_path)
