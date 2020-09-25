import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' 

import sys
sys.path.append('resnet_based')
sys.path.append('yolo_based')


import argparse

from PIL import Image

from resnet_based.ResnetFinetuneModel import ResnetFinetuneModel
from yolo_based.YoloFinetuneModel import YoloFinetuneModel

def resnet_recognition(rgb_path, depth_path):
    model = ResnetFinetuneModel('resnet_based/model/SUNRGBD_label2name_19.json')

    scene = model.test(rgb_path, depth_path, 'resnet_based/model/resnet_finetune_sunrgbd.h5')

    return scene

def yolo_recognition(rgb_path):
    model = YoloFinetuneModel('yolo_based/model_data/yolo4_anchors.txt', 'yolo_based/model_data/sunrgbd_object19scene19_100_classes.txt')

    scene = model.test(rgb_path, 'yolo_based/model_data/yolo_finetune_sunrgbd.h5')

    return scene

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-color', type=str, required=True)
    parser.add_argument('-depth', type=str, required=True)
    args = parser.parse_args()

    print(args.color, args.depth)

    scene = yolo_recognition(args.color)
    if(scene==None):
        scene = resnet_recognition(args.color, args.depth)
    print("result: ", scene)