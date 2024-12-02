import os

def generateRTMDETconfigFile(nms_pre, min_bbox_size, score_thr, max_per_img, rtmdet_dir="/mmdetection/configs/rtmdet_edited/"):
    template1 = open(os.path.join(rtmdet_dir, "template/template1.txt"), "r").readlines()
    template2 = open(os.path.join(rtmdet_dir, "template/template2.txt"), "r").readlines()
    config = [] 

    config += template1
    config.append("\n\t\tnms_pre=" + str(nms_pre) + ",\n")
    config.append("\t\tmin_bbox_size=" + str(min_bbox_size) + ",\n")
    config.append("\t\tscore_thr=" + str(score_thr) + ",\n")
    config.append("\t\tnms=dict(type='nms', iou_threshold=0.65),\n")
    config.append("\t\tmax_per_img=" + str(max_per_img) + "),\n")
    config += template2

    with open(os.path.join(rtmdet_dir, "rtmdet_l_8xb32-300e_coco.py"), "w") as output:
        output.writelines(config)

if __name__== "__main__":
    generateRTMDETconfigFile(nms_pre=10000, min_bbox_size=0, score_thr=0.3, max_per_img=100, rtmdet_dir="/media/neduchal/Data/Projekty/EdgeComputing/edge_computing_tests/configs/rtmdet/")
