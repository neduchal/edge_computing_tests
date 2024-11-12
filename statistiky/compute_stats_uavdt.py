import glob
import os
import re


def calculate_iou(box1, box2):
    # box1 a box2 jsou ve formátu [centerX, centerY, width, height]
    
    # Převod na souřadnice rohů [x1, y1, x2, y2]
    x1_box1 = box1[0] 
    y1_box1 = box1[1] 
    x2_box1 = box1[0] + box1[2]
    y2_box1 = box1[1] + box1[3] 
    
    x1_box2 = box2[0]
    y1_box2 = box2[1]
    x2_box2 = box2[0] + box2[2]
    y2_box2 = box2[1] + box2[3] 
    
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
            gt_data[line_data[0]].append([float(line_data[2]),float(line_data[3]),float(line_data[4]),float(line_data[5]),int(line_data[-1])])
        all_gt_data[dirname] = gt_data

pred_files = glob.glob("./data/output_uavdt/output/*.txt")



#pred_filename = "./data/output_uavdt/output/output_yolox_tiny.txt"

for pred_filename in pred_files:
    print(pred_filename)
    all_pred_data = {} 
    with open(pred_filename, "r") as pred_file:
        lines = pred_file.readlines()
        for line in lines:
            line_data = line.split(",")
            file_number = extract_number(line_data[1])
            if not all_pred_data.get(line_data[0]):
                all_pred_data[line_data[0]] = {}
            if not all_pred_data.get(line_data[0]).get(str(file_number)):
                all_pred_data.get(line_data[0])[str(file_number)] = []
            if (float(line_data[6]) > 0.5):
                all_pred_data.get(line_data[0]).get(str(file_number)).append([float(line_data[2]),float(line_data[3]),float(line_data[4]),float(line_data[5]), int(line_data[-1])])
            
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
            #print(folder, image_number)
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

    print (os.path.basename(pred_filename)[:-4], ": Spravne:", all_matched, "Vsechny:", all_total, "Recall:", all_matched/all_total, "Precision:", all_matched/pr_count, "Pred count", pr_count)

        
    