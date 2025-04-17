"""Definition of the VehicleDetector class."""
import os

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmcv.runner import wrap_fp16_model
import logging
import torch
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VehicleDetector:
    """Handles detection of vehicles on video frames."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = None):
        """Initialization of the Detector class.

        Args:
            config_path: Path to neural network configuration file
            checkpoint_path: Path to neural network checkpoint file
            device: Device on which the neural network makes inferences - 'cpu'/'cuda:0'.
                    Default value 'cuda:0' - first gpu available.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        if device is None or 'cuda' in device and torch.cuda.is_available():
            #device = self._choose_cuda()
            device = "cuda:0"
            logger.info(f'Device set to {device}')
        elif 'cuda' in device and not torch.cuda.is_available():
            logger.warning('Cuda selected as device and is not available. Falling back on CPU')
            device = 'cpu'
        self.model = init_detector(config_path, checkpoint_path, device=device)
        if os.getenv("DETECTOR_PRECISION", "fp32") == "fp16":
            wrap_fp16_model(self.model)
            logger.info("Detector model set to fp16 precision.")

    @staticmethod
    def _choose_cuda():
        """Choose cuda device based on available memory."""
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            memories = {}
            for i in range(count):
                mem = torch.cuda.mem_get_info(i)
                memories[i] = mem[0]  # total free memory on GPU
            device = max(memories, key=memories.get)
            return f'cuda:{device}'
        else:
            return 'cpu'

    def predict(
            self,
            image_batch: List[np.ndarray],
            filter_classes: List[int] = None
    ) -> List[List[Tuple[np.ndarray, int]]]:
        """Inference of the detector model.

        Args:
            image_batch: list of input images for detection
            filter_classes: list of classes numbers in predictions to preserve.
                            Default None - all classes are preserved.

        Returns:
            list of the same size as image_batch with detections for each input image. Detections are in format
                 ([[bbox], confidence], class)
        """
        if filter_classes is None:
            filter_classes = []

        # model inference
        predictions = inference_detector(self.model, image_batch)

        # class filtering based on coco class numbers
        output = []
        if len(filter_classes):
            for im in predictions:
                im_classes = []
                for val in filter_classes:
                    im_classes.append((im[val], val))
                output.append(im_classes)
        else:
            output = predictions

        return output


if __name__ == "__main__":
    config_file = r'/mmdetection/configs/yolox/yolox_l_8x8_300e_coco.py'
    checkpoint_file = \
        r'/root/detektory/modely/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'

    # bicycle, car, motorcycle, bus, train, truck as in coco specifications (adjusted to start from 0)
    class_filter = [1, 2, 3, 5, 6, 7]

    # same image just for test
    images = [cv2.imread(r'/root/data/Street View.v1i.coco/test/aguanambi-1000_png_jpg.rf.7179a0df58ad6448028bc5bc21dca41e.jpg')]

    car_detector = VehicleDetector(config_file, checkpoint_file)
    pred = car_detector.predict(images, filter_classes=class_filter)

    print("----------------")
    print(type(pred), len(pred))
    print("----------------")
    print(type(pred[0]), len(pred[0]))
    for i in range(len(pred[0])):
        print("------------")
        print(type(pred[0][i]), len(pred[0][i]))
        for j in range(len(pred[0][i])):
            print("---------")
            print(type(pred[0][i][j]))