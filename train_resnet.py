
import sys
sys.path.append('resnet_based')

from resnet_based.ResnetFinetuneModel import ResnetFinetuneModel



model = ResnetFinetuneModel('resnet_based/model/SUNRGBD_label2name_19.json')
model.train(
    'config/train.txt',
    'config/val.txt',
    'resnet_based/models/'
)