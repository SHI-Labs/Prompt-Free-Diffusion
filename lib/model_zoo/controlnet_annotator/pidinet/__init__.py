import os
import torch
import numpy as np
from einops import rearrange
from .model import pidinet

models_path = 'pretrained/controlnet/preprocess'

netNetwork = None
remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/table5_pidinet.pth"
modeldir = os.path.join(models_path, "pidinet")
old_modeldir = os.path.dirname(os.path.realpath(__file__))

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

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

def load_state_dict(ckpt_path, location='cpu'):
    def get_state_dict(d):
        return d.get('state_dict', d)

    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(
            ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def apply_pidinet(input_image, is_safe=False, apply_fliter=False, device='cpu'):
    global netNetwork
    if netNetwork is None:
        modelpath = os.path.join(modeldir, "table5_pidinet.pth")
        old_modelpath = os.path.join(old_modeldir, "table5_pidinet.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath
        elif not os.path.exists(modelpath):
            load_file_from_url(remote_model_path, model_dir=modeldir)
        netNetwork = pidinet()
        ckp = load_state_dict(modelpath)
        netNetwork.load_state_dict({k.replace('module.',''):v for k, v in ckp.items()})
        
    netNetwork = netNetwork.to(device)
    netNetwork.eval()
    assert input_image.ndim == 3
    input_image = input_image[:, :, ::-1].copy()
    with torch.no_grad():
        image_pidi = torch.from_numpy(input_image).float().to(device)
        image_pidi = image_pidi / 255.0
        image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
        edge = netNetwork(image_pidi)[-1]
        edge = edge.cpu().numpy()
        if apply_fliter:
            edge = edge > 0.5 
        if is_safe:
            edge = safe_step(edge)
        edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        
    return edge[0][0] 

def unload_pid_model():
    global netNetwork
    if netNetwork is not None:
        netNetwork.cpu()