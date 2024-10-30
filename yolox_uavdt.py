"""Definition of the VehicleDetector class."""
import os

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
import glob


def get_jpg_files(root_folder):
    jpg_files = []
    
    # Procházení složky UAV-benchmark-M
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        
        # Kontrola, jestli je složka ve formátu MXXXX
        if os.path.isdir(folder_path) and folder.startswith('M') and folder[1:].isdigit() and len(folder) == 5:
            # Procházení všech souborů v podsložce
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    jpg_files.append((os.path.join(folder_path, file), file, folder_path.split("/")[-1]))
    return jpg_files

def convert_yolox_to_yolov5(directory, image, im, threshold):
    yolov5_detections = []  # Výsledný seznam pro YOLOv5 výstup

    for val in range(len(im)):  # Nebo jakýkoli rozsah, který je relevantní
        pred_instances = im[val].pred_instances  # Získáš predikované instance

        bboxes = pred_instances.bboxes.cpu().numpy()  # Získáš souřadnice ohraničujících boxů (převedeme na numpy)
        scores = pred_instances.scores.cpu().numpy()  # Získáš skóre (confidence)
        labels = pred_instances.labels.cpu().numpy()  # Získáš třídy (labels)

        for bbox, score, label in zip(bboxes, scores, labels):
            class_filter = [1, 2, 3, 5, 6, 7]
            if score > threshold and label in class_filter:
                # Připravíš YOLOv5 formát: [0, 0, x_min, width, height, y_max, confidence, class]
                x_min, y_min, x_max, y_max = bbox
                yolov5_detections.append([directory, image, x_min, y_min, x_max-x_min, y_max-y_min, score, 0, 0, label])
    return yolov5_detections

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

config_file = r'/mmdetection/configs/yolox/yolox_x_8xb8-300e_coco.py'
checkpoint_file = r'~/detektory/modely/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        
model = init_detector(config_file, checkpoint_file, device="cuda")        
# same image just for test

files = get_jpg_files("/root/detektory/data/uavdt/UAV-benchmark-M/")

# Print iterations progress
l = len(files)
print("Pocet obrazku: ", l)

all_predictions = []

for i, f in enumerate(files):    
    images = [cv2.imread(f[0])]
    predictions = inference_detector(model, images)
    yolo5_pred = convert_yolox_to_yolov5(f[2], f[1], predictions, 0.5)
    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    all_predictions += yolo5_pred
    
print()
    
with open(os.path.join("/root/detektory/data/uavdt/output/output_yolox_x.txt"), "w") as fo:
    all_predictions_np = np.array(all_predictions)
    for item in all_predictions_np:
        fo.write(','.join(str(e) for e in item.tolist()) + "\n")  
        
         
