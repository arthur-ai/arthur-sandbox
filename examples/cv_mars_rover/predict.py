
from typing import List, Union

import torch
from tqdm import tqdm

# yolov5 does not import modules in a path-agnostic way
# workaround is to add the folder to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "yolov5"))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

from util.datasets import LoadImagesMulti


class MarsPredictor:

    @torch.no_grad()
    def __init__(self, weights_path, image_size: Union[List[int], int] = 1024, device=''):

        # set image size to [h, w]
        if not isinstance(image_size, list):
            image_size = [image_size, image_size]
        elif len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]

        # Initialize
        device = select_device(device)

        # Load model
        model = attempt_load(weights_path, map_location=device)
        stride = int(model.stride.max())  # model stride
        # TODO: use names?
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        image_size = check_img_size(image_size, s=stride)  # check image size

        self.image_size = image_size
        self.device = device
        self.model = model

    @torch.no_grad()
    def predict(self, image_paths, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000):
        if isinstance(image_paths, list) and len(image_paths) == 0:
            return []

        # create data loader
        stride = int(self.model.stride.max())  # model stride
        dataset = LoadImagesMulti(image_paths, img_size=self.image_size, stride=stride, auto=True)

        # Run inference
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.image_size).to(self.device).type_as(next(self.model.parameters())))  # run once

        # path to image, padded/preprocessed image, raw image
        results = []
        for path, img, im0s in tqdm(dataset, ncols=100):

            # load and process data
            img = torch.from_numpy(img).to(self.device)
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # get raw prediction from model
            pred = self.model(img, augment=False, visualize=False)[0]

            # non max suppression: gropu and filter inferences by confidence for single boxes
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            det = pred[0]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

            results.append(det.cpu().numpy())

        return results
