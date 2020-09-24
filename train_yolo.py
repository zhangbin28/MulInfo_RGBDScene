import sys
sys.path.append('yolo_based')

from yolo_based.YoloFinetuneModel import YoloFinetuneModel


model  = YoloFinetuneModel('yolo_based/model_data/yolo4_anchors.txt', 'yolo_based/model_data/sunrgbd_object19scene19_100_classes.txt')
model.train('config/train_obj.txt','config/val_obj.txt', 'yolo_based/models/')