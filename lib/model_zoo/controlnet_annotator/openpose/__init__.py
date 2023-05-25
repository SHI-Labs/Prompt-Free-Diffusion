# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
from . import util
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face

models_path = "pretrained/controlnet/preprocess"

from typing import NamedTuple, Tuple, List, Callable, Union

body_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
hand_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/hand_pose_model.pth"
face_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/facenet.pth"

HandResult = List[Keypoint]
FaceResult = List[Keypoint]

class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[HandResult, None]
    right_hand: Union[HandResult, None]
    face: Union[FaceResult, None]

def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas

def encode_poses_as_json(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> str:
    """ Encode the pose as a JSON string following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """
    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None
        
        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return json.dumps({
        'people': [
            {
                'pose_keypoints_2d': compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d":compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        'canvas_height': canvas_height,
        'canvas_width': canvas_width,
    }, indent=4)

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
    
class OpenposeDetector:
    """
    A class for detecting human poses in images using the Openpose model.

    Attributes:
        model_dir (str): Path to the directory where the pose models are stored.
    """
    model_dir = os.path.join(models_path, "openpose")

    def __init__(self, device):
        self.device = device
        self.body_estimation = None
        self.hand_estimation = None
        self.face_estimation = None

    def load_model(self):
        """
        Load the Openpose body, hand, and face models.
        """
        body_modelpath = os.path.join(self.model_dir, "body_pose_model.pth")
        hand_modelpath = os.path.join(self.model_dir, "hand_pose_model.pth")
        face_modelpath = os.path.join(self.model_dir, "facenet.pth")

        if not os.path.exists(body_modelpath):
            load_file_from_url(body_model_path, model_dir=self.model_dir)

        if not os.path.exists(hand_modelpath):
            load_file_from_url(hand_model_path, model_dir=self.model_dir)

        if not os.path.exists(face_modelpath):
            load_file_from_url(face_model_path, model_dir=self.model_dir)

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)
        self.face_estimation = Face(face_modelpath)

    def unload_model(self):
        """
        Unload the Openpose models by moving them to the CPU.
        """
        if self.body_estimation is not None:
            self.body_estimation.model.to("cpu")
            self.hand_estimation.model.to("cpu")
            self.face_estimation.model.to("cpu")

    def detect_hands(self, body: BodyResult, oriImg) -> Tuple[Union[HandResult, None], Union[HandResult, None]]:
        left_hand = None
        right_hand = None
        H, W, _ = oriImg.shape
        for x, y, w, is_left in util.handDetect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                
                hand_result = [
                    Keypoint(x=peak[0], y=peak[1])
                    for peak in peaks
                ]

                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result

        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg) -> Union[FaceResult, None]:
        face = util.faceDetect(body, oriImg)
        if face is None:
            return None
        
        x, y, w = face
        H, W, _ = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y+w, x:x+w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [
                Keypoint(x=peak[0], y=peak[1])
                for peak in peaks
            ]
        
        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """
        Detect poses in the given image.
            Args:
                oriImg (numpy.ndarray): The input image for pose detection.
                include_hand (bool, optional): Whether to include hand detection. Defaults to False.
                include_face (bool, optional): Whether to include face detection. Defaults to False.

        Returns:
            List[PoseResult]: A list of PoseResult objects containing the detected poses.
        """
        if self.body_estimation is None:
            self.load_model()
            
        self.body_estimation.model.to(self.device)
        self.hand_estimation.model.to(self.device)
        self.face_estimation.model.to(self.device)

        self.body_estimation.cn_device = self.device
        self.hand_estimation.cn_device = self.device
        self.face_estimation.cn_device = self.device

        oriImg = oriImg[:, :, ::-1].copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = (None,) * 3
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)
                
                results.append(PoseResult(BodyResult(
                    keypoints=[
                        Keypoint(
                            x=keypoint.x / float(W),
                            y=keypoint.y / float(H)
                        ) if keypoint is not None else None
                        for keypoint in body.keypoints
                    ], 
                    total_score=body.total_score,
                    total_parts=body.total_parts
                ), left_hand, right_hand, face))
            
            return results
        
    def __call__(
            self, oriImg, include_body=True, include_hand=False, include_face=False,
            json_pose_callback: Callable[[str], None] = None,
        ):
        """
        Detect and draw poses in the given image.

        Args:
            oriImg (numpy.ndarray): The input image for pose detection and drawing.
            include_body (bool, optional): Whether to include body keypoints. Defaults to True.
            include_hand (bool, optional): Whether to include hand keypoints. Defaults to False.
            include_face (bool, optional): Whether to include face keypoints. Defaults to False.
            json_pose_callback (Callable, optional): A callback that accepts the pose JSON string.

        Returns:
            numpy.ndarray: The image with detected and drawn poses.
        """
        H, W, _ = oriImg.shape
        poses = self.detect_poses(oriImg, include_hand, include_face)
        if json_pose_callback:
            json_pose_callback(encode_poses_as_json(poses, H, W))
        return draw_poses(poses, H, W, draw_body=include_body, draw_hand=include_hand, draw_face=include_face) 

class OpenposeModel(object):
    def __init__(self) -> None:
        self.model_openpose = None

    def run_model(
            self,
            img: np.ndarray,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            json_pose_callback: Callable[[str], None] = None,
            device = 'cpu', ):

        if json_pose_callback is None:
            json_pose_callback = lambda x: None

        if self.model_openpose is None:
            self.model_openpose = OpenposeDetector(device=device)

        return self.model_openpose(
            img,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
            json_pose_callback=json_pose_callback)

    def unload(self):
        if self.model_openpose is not None:
            self.model_openpose.unload_model()
