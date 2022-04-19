import cv2, json, keyboard, os, random, yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader

if __name__ == "__main__":
    modelName = 'faster_rcnn_R_50_FPN_3x'
    annotationName = 'Animal'
    version = 2

    configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', f'{modelName}_{annotationName}_v{version}', 'config.yaml')
    inferenceResultsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', f'{modelName}_{annotationName}_v{version}', 'inferenceResults.json')

    with open(configPath, 'r') as f:
        config = yaml.safe_load(f)
        nClasses = config['MODEL']['ROI_HEADS']['NUM_CLASSES']

    with open(inferenceResultsPath, 'r') as f:
        inferenceResults = json.load(f)
        print('Inference results:')
        for key in inferenceResults['bbox'].keys():
            if '-' not in key:
                print(f"\t{key}: {inferenceResults['bbox'][key]}")

    cfg = get_cfg()
    cfg.merge_from_file(configPath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = f'output/{modelName}_{annotationName}_v{version}/model_final.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nClasses
    predictor = DefaultPredictor(cfg)

    # dataDir = os.path.abspath(os.path.join(os.pardir, 'dataset'))
    # annTrain = os.path.join(dataDir, 'annotations', f'instances{annotationName}_train2017.json')
    # annVal = os.path.join(dataDir, 'annotations', f'instances{annotationName}_val2017.json')
    #
    # register_coco_instances("my_dataset_train", {}, annTrain, f'{dataDir}/images/train2017')
    # register_coco_instances("my_dataset_val", {}, annVal, f'{dataDir}/images/val2017')

    #----------------------------------------
    # dataset_dicts = DatasetCatalog.get("my_dataset_val")
    # my_dataset_test_metadata = MetadataCatalog.get("my_dataset_val")
    #
    # for i, d in enumerate(random.sample(dataset_dicts, 6)):
    #     im = cv2.imread(d["file_name"])
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1],
    #                    metadata=my_dataset_test_metadata,
    #                    scale=0.8,
    #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    #                    )
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imshow(str(i), v.get_image()[:, :, ::-1])
    # cv2.waitKey()
    #
    # print('Evaluating...')
    # evaluator = COCOEvaluator("my_dataset_val", output_dir=f"./output/")
    # val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    #
    # # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    # # predictor = DefaultPredictor(cfg)
    # #
    # # for i, d in enumerate(random.sample(dataset_dicts, 3)):
    # #     im = cv2.imread(d["file_name"])
    # #     outputs = predictor(im)
    # #     v = Visualizer(im[:, :, ::-1],
    # #                    metadata=my_dataset_train_metadata,
    # #                    scale=0.8,
    # #                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    # #                    )
    # #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # #     cv2.imshow(str(i), v.get_image()[:, :, ::-1])
    # # cv2.waitKey()
    # #
    # # evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
    # # val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    # # print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # #
    # # print(cfg.OUTPUT_DIR)
    # # cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "prueba")
    # # print(cfg.OUTPUT_DIR)
