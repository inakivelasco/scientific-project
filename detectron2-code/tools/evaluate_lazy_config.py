#!/usr/bin/env python
import logging
import os.path
import sys

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

import GPUtil, psutil, threading, time, cpuinfo, json
import cv2
import numpy as np

logger = logging.getLogger("detectron2")


def register_dataset():
    from detectron2.data.datasets import register_coco_instances

    # register_coco_instances("maize_train", {},
    #                         "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_train.json",
    #                         "/media/naeem/T7/datasets/maize_data_coco")
    register_coco_instances("maize_train", {},
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/train/trainNewAnnotations.json',
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/train/images')
    # register_coco_instances("maize_valid", {},
    #                         "/media/naeem/T7/datasets/maize_data_coco/annotations/instances_val.json",
    #                         "/media/naeem/T7/datasets/maize_data_coco")
    register_coco_instances("maize_valid", {},
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/test/testNewAnnotations.json',
                            'C:/Users/Inaki/Desktop/corn_dataset_v2/COCOFormat/test/images')

def do_test(cfg, model):
    ret = inference_on_dataset(
        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
    )

    print_csv_format(ret)
    return ret

def measure():
    global stop_thread
    t = threading.Timer(0.5, measure)
    t.start()

    if stop_thread:
        t.cancel()

    cpuPercent = psutil.cpu_percent()
    ramPercent = psutil.virtual_memory().percent

    GPUs = GPUtil.getGPUs()
    gpusPercent = []
    for i, gpu in enumerate(GPUs):
        gpusPercent.append(gpu.memoryUtil * 100)
    logging.getLogger('consumption').info(f'{cpuPercent} {ramPercent} {np.round(np.mean(gpusPercent), 1)}')


args = default_argument_parser().parse_args()
cfg = LazyConfig.load(args.config_file)
model_name = args.config_file.split('/')[-2]  # Assuming the config.yaml file is saved in a folder named as the model
# cfg.train.output_dir = "/media/naeem/T7/trainers/fcos_R_50_FPN_1x.py/output/"
cfg.train.output_dir = os.path.join(os.pardir, os.pardir, 'models', model_name, 'evaluation', 'test')
cfg.dataloader.test.num_workers = 0  # for debugging
# cfg = LazyConfig.apply_overrides(cfg, args.opts)
default_setup(cfg, args)
register_dataset()

logging.basicConfig(filename=os.path.join(os.pardir, os.pardir, 'models', model_name, 'evaluation', 'test',
                                          'consumptionLog.txt'),
                    filemode='w',
                    format='',
                    datefmt='',
                    level=logging.INFO)
loggerConsumption = logging.getLogger('consumption')
logging.getLogger('consumption').info(f'CPU RAM GPU')
stop_thread = False
thread = threading.Thread(target=measure)
thread.start()
print('Starting process, 10 seconds to start...')
time.sleep(10)

model = instantiate(cfg.model)
model.to(cfg.train.device)
# model = create_ddp_model(model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

do_test(cfg, model)

imagesDir = os.path.join(cfg.train.output_dir, 'images')
os.makedirs(imagesDir, exist_ok=True)

eval_loader = instantiate(cfg.dataloader.test)
model.eval()

class_labels = np.array(['Weeds', 'Maize', 'Bark'])  # HARDCODED
for class_label in class_labels:
    os.makedirs(os.path.join(imagesDir, class_label), exist_ok=True)
nImages2Save = 20
for idx, inputs in enumerate(eval_loader):
    if idx == nImages2Save:
        break
    # Images with predictions for every class
    outputs = model(inputs)
    outputs_mask = outputs[0]['instances']._fields['scores'] > 0.5
    image = cv2.imread(inputs[0]['file_name'])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    v = Visualizer(image[:, :, ::-1], scale=1.2)

    detected_classes = outputs[0]['instances'][outputs_mask]._fields['pred_classes'].tolist()
    detected_scores = outputs[0]['instances'][outputs_mask]._fields['scores'].tolist()
    detected_scores = [round(score, 2) for score in detected_scores]

    labels = [f"{class_labels[detected_classes[i]]} {detected_scores[i]}" for i in range(len(detected_classes))]
    out = v.overlay_instances(
        boxes=outputs[0]['instances'][outputs_mask]._fields['pred_boxes'].to("cpu").tensor.detach().numpy(),
        labels=labels)

    # cv2.imshow(f'image{idx}', out.get_image())
    cv2.imwrite(os.path.join(imagesDir, f'image{idx}.jpg'), out.get_image())
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print('hold')

    # Images with predictions for each class separately
    for classIdx in range(len(class_labels)):
        mask = [cls == classIdx for cls in detected_classes]

        v = Visualizer(image[:, :, ::-1], scale=1.2)
        out = v.overlay_instances(
            boxes=outputs[0]['instances'][outputs_mask][mask]._fields['pred_boxes'].to("cpu").tensor.detach().numpy(),
            labels=np.array(labels)[mask])

        cv2.imwrite(os.path.join(imagesDir, class_labels[classIdx], f'image{idx}.jpg'), out.get_image())

# Hardware specs
gpuNames = []
for gpu in GPUtil.getGPUs():
    gpuNames.append(gpu.name)
gpuNames = '\n        '.join(gpuNames)
hardwareSpecs = {
        'cpu': {
            'name': f'Model: {cpuinfo.get_cpu_info()["brand_raw"]}',
            'logical_cores': f'Total nº of logical cores: {psutil.cpu_count()}',
            'GHz': f'Speed: {np.round(psutil.cpu_freq().max/1000, 2)} GHz'
        },
        'ram': {
            'GB': f'Total memory: {np.round(psutil.virtual_memory().total / 1024 ** 3, 2)} GB'
        },
        'gpu': {
            'names': f'Models: {gpuNames}'
        }
    }
with open(os.path.join(cfg.train.output_dir, 'hardwareSpecs.json'), 'w+') as f:
    json.dump(hardwareSpecs, f)

print('Finishing process, 10 seconds left...')
time.sleep(10)
stop_thread = True
