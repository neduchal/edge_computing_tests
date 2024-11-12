import glob
import os
import json

def load_detections_gt(file_path):
    detections = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            if values[0] != 2:
                continue
            detections.append(values)  # [classID, centerX, centerY, width, height]
    return detections

def load_detections_pred_all():
    detections = {}
    with open("./data/output_deepstream/result_archive.txt", "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]

    for i in range(len(lines)):
        if lines[i][0] != "D":
            detections[lines[i]] = []
            results = json.loads(lines[i+1][9:].replace("'", '"'))
            detections[lines[i]] = results

    return detections

def convert_bbox_to_relative(bbox, image_width, image_height):
    # Rozložení hodnot (x, y, width, height)
    x, y, width, height = bbox
    
    # Přepočet na relativní (center_x, center_y, width, height)
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    rel_width = width / image_width
    rel_height = height / image_height
    
    return [center_x, center_y, rel_width, rel_height]

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

    # Funkce pro porovnání detekcí
def compare_detections(groundtruth, predictions, iou_threshold=0.5):
    print(groundtruth)

    print(predictions)

    matched_detections = 0
    total_groundtruth = len(groundtruth)
    
    for gt_box in groundtruth:
        for pred_box in predictions:
            iou = calculate_iou(gt_box[1:], convert_bbox_to_relative(pred_box["bbox"], 960, 544))  # Porovnáváme souřadnice [centerX, centerY, width, height]
            if iou >= iou_threshold:
                matched_detections += 1
                break

    return matched_detections, total_groundtruth

files = glob.glob(os.path.join("./data/datasety/archive/test/images/", "*.jpg"))
all_matched = 0
all_total = 0
gt_count = 0
pr_count = 0

detections = load_detections_pred_all()

for fn in files:
    print(fn)
    detections_gt = load_detections_gt(os.path.join("./data/datasety/archive/test/labels/", os.path.basename(fn)[:-3] + "txt"))
    detection = detections.get(os.path.basename(fn))

    
    matched, total = compare_detections(detections_gt, detection, iou_threshold=0.5)
  
    all_matched += matched
    all_total += total
    gt_count += len(detections_gt)
    pr_count += len(detection)
    
print ("Spravne:", all_matched, "Vsechny:", all_total, "Recall:", all_matched/all_total, "Precision:", all_matched/pr_count)