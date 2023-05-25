import cv2
import numpy as np
import torch
import os

from einops import rearrange
from .models.mbv2_mlsd_tiny import  MobileV2_MLSD_Tiny
from .models.mbv2_mlsd_large import  MobileV2_MLSD_Large
from .utils import  pred_lines

models_path = 'pretrained/controlnet/preprocess'

mlsdmodel = None
remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"
old_modeldir = os.path.dirname(os.path.realpath(__file__))
modeldir = os.path.join(models_path, "mlsd")

def unload_mlsd_model():
    global mlsdmodel
    if mlsdmodel is not None:
        mlsdmodel = mlsdmodel.cpu()

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    from torch.hub import download_url_to_file, get_dir
    from urllib.parse import urlparse
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def apply_mlsd(input_image, thr_v, thr_d, device='cpu'):
    global modelpath, mlsdmodel
    if mlsdmodel is None:
        modelpath = os.path.join(modeldir, "mlsd_large_512_fp32.pth")
        old_modelpath = os.path.join(old_modeldir, "mlsd_large_512_fp32.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath
        elif not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=modeldir)
        mlsdmodel = MobileV2_MLSD_Large()
        mlsdmodel.load_state_dict(torch.load(modelpath), strict=True)
    mlsdmodel = mlsdmodel.to(device).eval()
        
    model = mlsdmodel
    assert input_image.ndim == 3
    img = input_image
    img_output = np.zeros_like(img)
    try:
        with torch.no_grad():
            lines = pred_lines(img, model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1)
    except Exception as e:
        pass
    return img_output[:, :, 0]
