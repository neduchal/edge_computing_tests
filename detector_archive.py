"""Definition of the VehicleDetector class."""
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import logging
import torch
from typing import List, Tuple
import glob
import os

logger = logging.getLogger(__name__)


def convert_yolox_to_yolov5(img_width, img_height, yolox_output, threshold):
    yolov5_output = []

    for detection_group in yolox_output:
        detections, class_id = detection_group
        for det in detections:
            # Extract YOLOX format bounding box and confidence
            min_x, min_y, max_x, max_y, confidence = det
            
            # Convert to YOLOv5 format: (x_center, y_center, width, height)
            x_center = (min_x + max_x) / 2 / img_width
            y_center = (min_y + max_y) / 2 / img_height
            width = (max_x - min_x) /img_width
            height = (max_y - min_y) / img_height
            
            # Append to YOLOv5 output: [class_id, x_center, y_center, width, height, confidence]
            if (threshold < confidence):
                yolov5_output.append([class_id, x_center, y_center, width, height])
            
    return np.array(yolov5_output)

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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif 'cuda' in device and not torch.cuda.is_available():
            logger.warning('Cuda selected as device and is not available. Falling back on CPU')
            device = 'cpu'
        self.model = init_detector(config_path, checkpoint_path, device=device)

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
    config_file = './VehicleDetector/yolox_x_8x8_300e_coco.py'
    checkpoint_file = './VehicleDetector/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'

    # bicycle, car, motorcycle, bus, train, truck as in coco specifications (adjusted to start from 0)
    class_filter = [1, 2, 3, 5, 6, 7]

    # same image just for test
    files = glob.glob("/home/jetson/edge/archive/test/images/*.jpg")
    car_detector = VehicleDetector(config_file, checkpoint_file)

    for f in files:
        print(f)
        images = [cv2.imread(f)]
        pred = car_detector.predict(images, filter_classes=class_filter)
        yolo5_pred = convert_yolox_to_yolov5(images[0].shape[1], images[0].shape[0],pred[0], 0.5)
        with open(os.path.join("/home/jetson/edge/archive/test/output_x/", os.path.basename(f) + ".txt"), "w") as fo:
            for item in yolo5_pred:
                fo.write(' '.join(str(e) for e in item.tolist()) + "\n")   
