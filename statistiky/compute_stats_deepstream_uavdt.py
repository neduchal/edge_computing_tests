import glob
import os
import re
import json

def load_detections_pred_all():
    detections = {}
    with open("./data/output_deepstream/results_uavdt.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

        for i in range(len(lines)):
            if lines[i][0] == "/":
                filepath = lines[i].split("/")
                dirname = filepath[-2]
                filename = filepath[-1]
                file_number = extract_number(filename)
                if (not detections.get(dirname)):
                    detections[dirname] = {}
                if (not detections.get(dirname).get(str(file_number))):
                    detections[dirname][str(file_number)] = []

                results = json.loads(lines[i+1][9:].replace("'", '"'))
                for r in results:
                    detections[dirname][str(file_number)].append(convert_bbox_to_relative(r["bbox"], 960, 544))
                detections[dirname]
    return detections

def calculate_iou(box1, box2):
    # box1 a box2 jsou ve formátu [centerX, centerY, width, height]
    
    # Převod na souřadnice rohů [x1, y1, x2, y2]
    x1_box1 = box1[0] - box1[2] / 2
    y1_box1 = box1[1] - box1[3] / 2
    x2_box1 = box1[0] + box1[2] / 2
    y2_box1 = box1[1] + box1[3] / 2
    
    x1_box2 = box2[0] - box2[2] / 2
    y1_box2 = box2[1] - box2[3] / 2
    x2_box2 = box2[0] + box2[2] / 2
    y2_box2 = box2[1] + box2[3] / 2
    
    # Výpočet souřadnic průniku
    x1_inter = max(x1_box1, x1_box2)
    y1_inter = max(y1_box1, y1_box2)
    x2_inter = min(x2_box1, x2_box2)
    y2_inter = min(y2_box1, y2_box2)
    
    # Rozměry průniku
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Výpočet ploch obou obdélníků
    box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
    
    # Výpočet IoU
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def compare_detections(groundtruth, predictions, iou_threshold=0.1):
    matched_detections = 0
    total_groundtruth = len(groundtruth)
    
    for gt_box in groundtruth:
        for pred_box in predictions:
            iou = calculate_iou(gt_box, pred_box)  # Porovnáváme souřadnice [centerX, centerY, width, height]
            if iou >= iou_threshold:
                matched_detections += 1
                break
    
    return matched_detections, total_groundtruth

def extract_number(filename):
    match = re.search(r'img0*(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return None

def convert_bbox_to_relative(bbox, image_width, image_height):
    # Rozložení hodnot (x, y, width, height)
    x, y, width, height = bbox
    
    # Přepočet na relativní (center_x, center_y, width, height)
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    rel_width = width / image_width
    rel_height = height / image_height
    
    return [center_x, center_y, rel_width, rel_height]

gt_files = glob.glob("./data/output_uavdt/UAV-benchmark-MOTD_v1.0/GT/*_gt_whole.txt")

all_gt_data = {}

for gt_file in gt_files:
    dirname = os.path.basename(gt_file).split("_")[0]
    with open(gt_file, "r") as f:
        lines = f.readlines()  
        gt_data = {}
        for line in lines:
            line_data = line.split(",")
            if (not gt_data.get(line_data[0])):
                gt_data[line_data[0]] = []
            gt_data[line_data[0]].append(convert_bbox_to_relative([float(line_data[2]),float(line_data[3]),float(line_data[4]),float(line_data[5])], 1024, 544))
        all_gt_data[dirname] = gt_data

pred_files = glob.glob("./data/output_uavdt/output/*.txt")


all_pred_data = load_detections_pred_all()

""" print(all_pred_data["M1305"]["1"])
print()
print(all_gt_data["M1305"]["1"]) """

all_matched = 0
all_total = 0
gt_count = 0
pr_count = 0

for folder in all_gt_data:
    for image_number in all_gt_data.get(folder):
        gt_image_data =  all_gt_data.get(folder).get(image_number)
        matched = 0
        total = len(gt_image_data)

        if (all_pred_data.get(folder)):
            if (all_pred_data.get(folder).get(image_number)):      

                pred_image_data = all_pred_data.get(folder).get(image_number)
                matched, total = compare_detections(gt_image_data, pred_image_data, iou_threshold=0.5)
                pr_count += len(pred_image_data)
        all_matched += matched
        all_total += total
        gt_count += len(gt_image_data)

print ("Spravne:", all_matched, "Vsechny:", all_total, "Recall:", all_matched/all_total, "Precision:", all_matched/pr_count, "Pred count", pr_count)

        
    