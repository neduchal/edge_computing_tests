import glob
import os
import numpy as np

def load_yolo_detections_gt(file_path):
    detections = {}
    detections["filename"] = os.path.basename(file_path)[:-3] + "jpg"
    detections["predictions"] = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            if values[0] != 2:
                continue
            detections["predictions"].append(values)  # [classID, centerX, centerY, width, height]
    return detections

def load_predictions(file_path):
    predictions = {}
    with open(file_path, "r") as input:
        lines = input.readlines()
        for line in lines:
            line_arr = line.strip().split(" ")
            nms_pre = line_arr[0]
            min_bbox_size = line_arr[1]
            score_thr = line_arr[2]
            max_per_img = line_arr[3]
            rtmdet_config_filename = line_arr[4]
            img_filename = line_arr[5]
            img_time = line_arr[6]
            prediction = list(map(float, line_arr[7:]))

            if predictions.get(str(nms_pre)) == None:
                predictions[str(nms_pre)] = {}
            if predictions.get(str(nms_pre)).get(str(min_bbox_size)) == None:
                predictions.get(str(nms_pre))[str(min_bbox_size)] = {}
            if predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr)) == None:
                predictions.get(str(nms_pre)).get(str(min_bbox_size))[str(score_thr)] = {}
            if predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr)).get(str(max_per_img)) == None:
                predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr))[str(max_per_img)] = {}   
            if predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr)).get(str(max_per_img)).get(str(rtmdet_config_filename)) == None:
                predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr)).get(str(max_per_img))[str(rtmdet_config_filename)]  = {}
            predictions_for_net = predictions.get(str(nms_pre)).get(str(min_bbox_size)).get(str(score_thr)).get(str(max_per_img)).get(str(rtmdet_config_filename))
            if predictions_for_net.get(str(img_filename)) == None:
                predictions_for_net[str(img_filename)] = {}
            if predictions_for_net.get(str(img_filename)).get("time") == None:
                predictions_for_net.get(str(img_filename))["time"] = img_time
            if predictions_for_net.get(str(img_filename)).get("predictions") == None:    
                predictions_for_net.get(str(img_filename))["predictions"] = [] 
            predictions_for_net.get(str(img_filename)).get("predictions").append(prediction)
    return predictions



def load_yolo_detections(file_path):
    detections = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            detections.append(values)  # [classID, centerX, centerY, width, height]
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

# Funkce pro porovnání detekcí
def compare_detections(groundtruth, predictions, iou_threshold=0.5):
    matched_detections = 0
    total_groundtruth = len(groundtruth)
    
    for gt_box in groundtruth:
        for pred_box in predictions:
            iou = calculate_iou(gt_box[1:], pred_box[1:])  # Porovnáváme souřadnice [centerX, centerY, width, height]
            if iou >= iou_threshold:
                matched_detections += 1
                break

    return matched_detections, total_groundtruth


predictions = load_predictions("/media/neduchal/bigDisk/edgecomputing/datasety/output/rtmdet.txt")

#print(predictions.keys(), predictions["100"].keys(), predictions["100"]["0"]["0.1"]["10"]["rtmdet_x_8xb32-300e_coco.py"]["screenshot_36369_jpg.rf.6ef56409df9f8d5ce61903b230a84fce.jpg"]["predictions"])

files = glob.glob(os.path.join("/media/neduchal/bigDisk/edgecomputing/datasety/Street View.v1i.yolov5pytorch/test/labels/", "*.txt"))

gt = [] 
for fn in files:
    gt.append(load_yolo_detections_gt(fn))

output_file = open("rtmdet_stats.txt", "w")
output_file.write("nms_pre,min_bbox_size,score_thr,max_per_img,network,avg_time,precision,recall, F1;\n")

for nms_pre in predictions.keys():
    for min_bbox_size in predictions[nms_pre].keys():
        for score_thr in predictions[nms_pre][min_bbox_size].keys():
            for max_per_img in predictions[nms_pre][min_bbox_size][score_thr].keys(): 
                for network in predictions[nms_pre][min_bbox_size][score_thr][max_per_img].keys(): 
                    predictions_for_network = predictions[nms_pre][min_bbox_size][score_thr][max_per_img][network]
                    times = []

                    all_matched = 0
                    all_total = 0
                    gt_count = 0
                    pr_count = 0
                    for img in gt:
                        predictions_img_gt = img["predictions"]
                        if predictions_for_network.get(img["filename"]) == None:
                            predictions_img = []
                        else:
                            predictions_img = predictions_for_network[img["filename"]]["predictions"]
                            times.append(float(predictions_for_network[img["filename"]]["time"]))
                        matched, total = compare_detections(predictions_img_gt, predictions_img, iou_threshold=0.3)        
                        all_matched += matched
                        all_total += total
                        gt_count += len(predictions_img_gt)
                        pr_count += len(predictions_img)

                    recall = all_matched/all_total
                    precision = all_matched/pr_count
                    F1 = (2 * precision*recall)/(precision + recall)
                    output_file.write(",".join([nms_pre, min_bbox_size, score_thr, max_per_img, network, str(np.mean(times)), str(precision), str(recall), str(F1)]) + ";\n")

output_file.close()



""" 
files = glob.glob(os.path.join("./data/datasety/archive/test/images/", "*.jpg"))
all_matched = 0
all_total = 0
gt_count = 0
pr_count = 0

for fn in files:
    #print(fn)
    detections = load_yolo_detections(os.path.join("./data/output/output_rtmdet_s/", os.path.basename(fn) + ".txt"))
    detections_gt = load_yolo_detections_gt(os.path.join("./data/datasety/archive/test/labels/", os.path.basename(fn)[:-3] + "txt"))
    
    matched, total = compare_detections(detections_gt, detections, iou_threshold=0.5)
    all_matched += matched
    all_total += total
    gt_count += len(detections_gt)
    pr_count += len(detections)

recall = all_matched/all_total
precision = all_matched/pr_count
F1 = (2 * precision*recall)/(precision + recall)
    
print ("Spravne:", all_matched, "Vsechny:", all_total, "Recall:", all_matched/all_total, "Precision:", all_matched/pr_count, "F1:",  F1)

         """
        