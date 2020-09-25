# Mul-info_rgbd-scene

This project provides an implementation for paper "An Efﬁcient RGB-D Scene Recognition Method based on Multi-information Fusion".

## Usage

Get trained model and resnet/yolo4 model
```
cd resnet_based/model
wget ∰
wget ∰

cd ../../yolo_based/model_data
wget ∰
wget ∰
```

Run
```
python3 demo.py -color path2rgb_image -depth path2depth_image
```

## Train

### Resnet-based
Label file
```
<rgb_image> <depth_image> <scene>
```

### Yolo-based
Label file
```
<rgb_image> <object> (<boject> <boject> ...)
<object> => <x> <y> <width> <height> <object@scene>
```

### Example
The example label files is shown in directory config/
```
resnet_based:   train.txt val.txt
yolo_based:     train_obj.txt val_obj.txt
```






