"""Definition of the VehicleDetector class."""
import os

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import glob
import time
from common import generateRTMDETconfigFile

def convert_yolox_to_yolov5(img_width, img_height, im, threshold, nms_pre, min_bbox_size, score_thr, max_per_img, rtmdet_config_filename, img_filename, avg_time):
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
                yolov5_detections.append([nms_pre, min_bbox_size, score_thr, max_per_img, rtmdet_config_filename, img_filename, avg_time, label, rel_cx, rel_cy, rel_width, rel_height, score])
    return np.array(yolov5_detections)

nms_pre_arr = [100, 250, 500, 1000, 2000]
min_bbox_size_arr = [0, 10, 25]
score_thr_arr = [0.1, 0.3, 0.5]
max_per_img_arr = [10, 25, 50]

""" nms_pre_arr = [100, 250]
min_bbox_size_arr = [0,]
score_thr_arr = [0.5]
max_per_img_arr = [10]

config_files = ["rtmdet_s_8xb32-300e_coco.py", 
                "rtmdet_tiny_8xb32-300e_coco.py"] 
checkpoint_files = ["rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth", 
                    "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"] """


config_files = ["rtmdet_x_8xb32-300e_coco.py", 
                "rtmdet_l_8xb32-300e_coco.py", 
                "rtmdet_m_8xb32-300e_coco.py", 
                "rtmdet_s_8xb32-300e_coco.py", 
                "rtmdet_tiny_8xb32-300e_coco.py"] 
checkpoint_files = ["rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
                    "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth", 
                    "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth", 
                    "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth", 
                    "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"] 


rtmdet_config_dir="/mmdetection/configs/rtmdet_edited/" 
rtmdet_checkpoint_dir="~/detektory/modely/"

files = glob.glob("/root/data/Street View.v1i.yolov5pytorch/test/images/*.jpg")

output_file = open("/root/data/output/rtmdet.txt", "w")

for nms_pre in nms_pre_arr:
    for min_bbox_size in min_bbox_size_arr:
        for score_thr in score_thr_arr:
            for max_per_img in max_per_img_arr:
                generateRTMDETconfigFile(nms_pre=nms_pre, min_bbox_size=min_bbox_size, score_thr=score_thr, max_per_img=max_per_img, rtmdet_dir="/mmdetection/configs/rtmdet_edited/")
                for config_file, checkpoint_file in zip(config_files, checkpoint_files):
                    print("Settings:", nms_pre, min_bbox_size, score_thr, max_per_img, config_file)
                    model = init_detector(os.path.join(rtmdet_config_dir, config_file), os.path.join(rtmdet_checkpoint_dir, checkpoint_file), device="cuda")        
                    times = []
                    for f in files:   
                        images = [cv2.imread(f)]
                        start = time.time()
                        predictions = inference_detector(model, images)
                        times.append(time.time() - start)
                        yolo5_pred = convert_yolox_to_yolov5(images[0].shape[1], images[0].shape[0], predictions, 0.5, nms_pre, min_bbox_size, score_thr, max_per_img, config_file, os.path.basename(f), times[-1])
                        for item in yolo5_pred:
                            output_file.write(' '.join(str(e) for e in item.tolist()) + "\n")  

output_file.close()                           
#print(np.mean(times))