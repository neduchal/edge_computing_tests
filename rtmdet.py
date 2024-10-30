"""Definition of the VehicleDetector class."""
import os

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import glob
import time

def convert_yolox_to_yolov5(img_width, img_height, im, threshold):
    yolov5_detections = []  # Výsledný seznam pro YOLOv5 výstup

    for val in range(len(im)):  # Nebo jakýkoli rozsah, který je relevantní
        pred_instances = im[val].pred_instances  # Získáš predikované instance

        bboxes = pred_instances.bboxes.cpu().numpy()  # Získáš souřadnice ohraničujících boxů (převedeme na numpy)
        scores = pred_instances.scores.cpu().numpy()  # Získáš skóre (confidence)
        labels = pred_instances.labels.cpu().numpy()  # Získáš třídy (labels)

        for bbox, score, label in zip(bboxes, scores, labels):
            class_filter = [1, 2, 3, 5, 6, 7]
            if score > threshold and label in class_filter:
                # Připravíš YOLOv5 formát: [cx, cy, width, height, confidence, class]
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                cx = (x_max + x_min) / 2
                cy = (y_max + y_min) / 2
                rel_width = width / img_width
                rel_height = height / img_height
                rel_cx = cx / img_width
                rel_cy = cy / img_height
                yolov5_detections.append([label, rel_cx, rel_cy, rel_width, rel_height, score])
    return np.array(yolov5_detections)

config_file = r'/mmdetection/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py'
checkpoint_file = r'~/detektory/modely/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
        
model = init_detector(config_file, checkpoint_file, device="cuda")        
# same image just for test
files = glob.glob("/root/detektory/data/archive/test/images/*.jpg")
times = []

for f in files:   
    images = [cv2.imread(f)]
    start = time.time()
    predictions = inference_detector(model, images)
    times.append(time.time() - start)
    yolo5_pred = convert_yolox_to_yolov5(images[0].shape[1], images[0].shape[0], predictions, 0.5)
    #with open(os.path.join("/root/detektory/data/archive/test/output_rtmdet_tiny/", os.path.basename(f) + ".txt"), "w") as fo:
    #    for item in yolo5_pred:
    #        fo.write(' '.join(str(e) for e in item.tolist()) + "\n")   

print(np.mean(times))