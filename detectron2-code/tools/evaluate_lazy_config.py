#!/usr/bin/env python
import logging
import os.path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import cv2
import numpy as np

logger = logging.getLogger("detectron2")

def register_dataset():
    from detectron2.data.datasets import register_coco_instances

    # register_coco_instances("maize_train", {},
    #                         "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_train.json",
    #                         "/media/naeem/T7/datasets/maize_data_coco")
    # register_coco_instances("maize_train", {},
    #                         "/../../../Scientific-Project/dataset/annotations/instancesAnimal_train2017.json",
    #                         "/../../../Scientific-Project/dataset/images/train2017")
    register_coco_instances("maize_train", {},
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/train/trainNewAnnotations.json',
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/train/images')
    # register_coco_instances("maize_valid", {},
    #                         "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
    #                         "/media/naeem/T7/datasets/maize_data_coco")
    # register_coco_instances("maize_valid", {},
    #                         "/../../../Scientific-Project/dataset/annotations/instancesAnimal_val2017.json",
    #                         "/../../../Scientific-Project/dataset/images/val2017")
    # register_coco_instances("maize_valid", {},
    #                         'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/val/valNewAnnotations.json',
    #                         'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/val/images')
    register_coco_instances("maize_valid", {},
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/test/testNewAnnotations.json',
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/test/images')

def do_test(cfg, model):
    ret = inference_on_dataset(
        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    )
    print_csv_format(ret)
    return ret

args = default_argument_parser().parse_args()
# cfg_file = "../configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py"
# cfg = LazyConfig.load(cfg_file)
cfg = LazyConfig.load(args.config_file)  # Added by inaki, Default the two lines above
# trained_iter = int(cfg.train.init_checkpoint.split('-')[-1].split('/')[0])  # Added by inaki
model_name = args.config_file.split('/')[-2]
cfg.train.output_dir = f"/Scientific-Project/models/{model_name}/evaluation/val"  # Default "/media/naeem/T7/trainers/fcos_R_50_FPN_1x.py/output/"
cfg.train.output_dir = f"/Scientific-Project/models/{model_name}/evaluation/test"  # Default "/media/naeem/T7/trainers/fcos_R_50_FPN_1x.py/output/"
cfg.dataloader.test.num_workers = 0 # for debugging
# cfg = LazyConfig.apply_overrides(cfg, args.opts)
default_setup(cfg, args)
register_dataset()

model = instantiate(cfg.model)
model.to(cfg.train.device)
# model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

do_test(cfg, model)  # Added by inaki

# Added by inaki
imagesDir = os.path.join(cfg.train.output_dir, 'images')
os.makedirs(imagesDir, exist_ok=True)

eval_loader = instantiate(cfg.dataloader.test)
model.eval()
# class_labels = np.array(['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'])  # Added by inaki, hardcoded
class_labels = np.array(['Weeds', 'Maize', 'Bark'])  # Added by inaki, hardcoded
for class_label in class_labels:
    os.makedirs(os.path.join(imagesDir, class_label), exist_ok=True)
nImages2Save = 20
for idx, inputs in enumerate(eval_loader):
    outputs = model(inputs)
    outputs_mask = outputs[0]['instances']._fields['scores'] > 0.5  # Added by inaki
    image = cv2.imread(inputs[0]['file_name'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    v = Visualizer(image[:, :, ::-1], scale=1.2)
    # out = v.draw_instance_predictions(outputs[0]['instances'][outputs_mask].to('cpu'))

    detected_classes = outputs[0]['instances'][outputs_mask]._fields['pred_classes'].tolist()  # Added by inaki
    detected_scores = outputs[0]['instances'][outputs_mask]._fields['scores'].tolist()  # Added by inaki
    detected_scores = [round(score, 2) for score in detected_scores]

    labels = [f"{class_labels[detected_classes[i]]} {detected_scores[i]}" for i in range(len(detected_classes))]
    out = v.overlay_instances(boxes=outputs[0]['instances'][outputs_mask]._fields['pred_boxes'].to("cpu").tensor.detach().numpy(),
                              labels=labels)  # Added by inaki, Default v.draw_instance_predictions

    # cv2.imshow(f'image{idx}', out.get_image())
    cv2.imwrite(os.path.join(imagesDir, f'image{idx}.jpg'), out.get_image())
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print('hold')

    # Images with predictions for each class separately
    for classIdx in range(len(class_labels)):
        mask = [cls == classIdx for cls in detected_classes]

        labels = [f"{class_labels[detected_classes[i]]} {detected_scores[i]}" for i in range(len(detected_classes))]
        v = Visualizer(image[:, :, ::-1], scale=1.2)
        out = v.overlay_instances(
            boxes=outputs[0]['instances'][outputs_mask][mask]._fields['pred_boxes'].to("cpu").tensor.detach().numpy(),
            labels=np.array(labels)[mask])  # Added by inaki, Default v.draw_instance_predictions

        cv2.imwrite(os.path.join(imagesDir, class_labels[classIdx], f'image{idx}.jpg'), out.get_image())

    nImages2Save -= 1
    if nImages2Save == 0:
        break

