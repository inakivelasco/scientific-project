import cv2, gc, json, os, random, torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch:", TORCH_VERSION, "; cuda:", CUDA_VERSION)
    print("cuda available:", torch.cuda.is_available(), " ", torch.cuda.get_device_name(0))

    # Model specification
    modelName = 'faster_rcnn_R_50_FPN_3x'
    annotationName = 'Animal'  # Only specific classes instead of all COCO dataset classes
    version = 99

    dataDir = os.path.abspath(os.path.join(os.pardir, 'dataset'))
    annTrain = os.path.join(dataDir, 'annotations', f'instances{annotationName}_train2017.json')
    annVal = os.path.join(dataDir, 'annotations', f'instances{annotationName}_val2017.json')

    register_coco_instances("my_dataset_train", {}, annTrain, f'{dataDir}/images/train2017')
    register_coco_instances("my_dataset_val", {}, annVal, f'{dataDir}/images/val2017')

    # Model parameters & train
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{modelName}.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.TEST.EVAL_PERIOD = 0  # default 0

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{modelName}.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.GAMMA = 0.1

    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = ()
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'{modelName}_{annotationName}_v{version}')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), "w") as f:
        f.write(cfg.dump())

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference & evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(inference)
    with open(os.path.join(cfg.OUTPUT_DIR, 'inferenceResults.json'), 'w') as f:
        json.dump(inference, f)
