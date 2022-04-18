import cv2, gc, json, os, random, torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("cuda available: ", torch.cuda.is_available(), " ", torch.cuda.get_device_name(0))

    # Model specification
    modelName = 'faster_rcnn_R_50_FPN_3x'
    annotationName = 'Animal'
    version = 1

    dataDir = os.path.abspath(os.path.join(os.pardir, 'dataset'))
    annTrain = os.path.join(dataDir, 'annotations', f'instances{annotationName}_train2017.json')
    annVal = os.path.join(dataDir, 'annotations', f'instances{annotationName}_val2017.json')

    register_coco_instances("my_dataset_train", {}, annTrain, f'{dataDir}/images/train2017')
    register_coco_instances("my_dataset_val", {}, annVal, f'{dataDir}/images/val2017')

    # Verify data loading
    my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow(str(i), vis.get_image()[:, :, ::-1])
    # cv2.waitKey()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{modelName}.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{modelName}.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001  # (previous value: 0.00025) pick a good LR

    # cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = ()       # do not decay learning rate, or when to decay it?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # (previous value: 128) faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'{modelName}_{annotationName}_v{version}')
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, 'config.yaml'), "w") as f:
        f.write(cfg.dump())

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Inference and evaluation
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get("my_dataset_val")
    my_dataset_metadata = MetadataCatalog.get("my_dataset_val")
    for i, d in enumerate(random.sample(dataset_dicts, 3)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=my_dataset_metadata,
                       scale=0.8
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow(f'image {i}', out.get_image()[:, :, ::-1])
    # cv2.waitKey()

    evaluator = COCOEvaluator("my_dataset_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(inference)
    with open(os.path.join(cfg.OUTPUT_DIR, 'inferenceResults.json'), 'w') as f:
        json.dump(inference, f)
