import json, os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from detectron2.config import get_cfg
from pycocotools.cocoeval import COCOeval
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog

if __name__ == "__main__":
    # Model specification -----
    modelName = 'faster_rcnn_R_50_FPN_3x'
    annotationName = 'Animal'
    version = 2

    # Prepare directories -----
    modelDir = os.path.abspath(os.path.join(os.pardir, 'models', f'{modelName}_{annotationName}_v{version}'))
    dataDir = os.path.abspath(os.path.join(os.pardir, 'dataset'))
    annValDir = os.path.join(dataDir, 'annotations', f'instances{annotationName}_val2017.json')

    testDir = os.path.join(modelDir, 'test')
    os.makedirs(testDir, exist_ok=False)
    setup_logger(output=os.path.join(testDir, 'testLog.txt'))  # TODO add this to train

    plotDir = os.path.join(modelDir, 'plots')
    os.makedirs(plotDir, exist_ok=False)

    register_coco_instances("my_dataset_val", {}, annValDir, f'{dataDir}/images/val2017')  # Evaluating with val dataset

    # Load model and config & evaluator and loader -----
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(modelDir, 'config.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(modelDir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=testDir)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")

    # Graphs pre-inference -----
    with open(os.path.join(modelDir, 'metrics.json'), 'r') as f:
        metrics = [json.loads(line) for line in f]
        iterations = [iterData['iteration'] for iterData in metrics]

        # Total loss / iter
        totalLoss = [iterData['total_loss'] for iterData in metrics]
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, totalLoss)
        plt.title('Training total_loss')
        plt.xlabel('Iteration')
        plt.ylabel('Total loss')
        plt.savefig(os.path.join(plotDir, 'totalLossPerIter.png'), bbox_inches='tight')

        # Loss box reg / iter
        lossBoxReg = [iterData['loss_box_reg'] for iterData in metrics]
        plt.clf()
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, lossBoxReg)
        plt.title('Training loss_box_reg')
        plt.xlabel('Iteration')
        plt.ylabel('Loss box reg')
        plt.savefig(os.path.join(plotDir, 'lossBoxRegPerIter.png'), bbox_inches='tight')

        # Loss box reg / iter
        lossCls = [iterData['loss_cls'] for iterData in metrics]
        plt.clf()
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, lossCls)
        plt.title('Training loss_cls')
        plt.xlabel('Iteration')
        plt.ylabel('Loss cls')
        plt.savefig(os.path.join(plotDir, 'lossClsPerIter.png'), bbox_inches='tight')

        # FPS / iter
        fps = [cfg.SOLVER.IMS_PER_BATCH / iterData['time'] for iterData in metrics]
        plt.clf()
        plt.figure(figsize=(15, 9))
        plt.plot(iterations, fps, label='FPS/iter')
        plt.plot(iterations, [np.mean(fps)] * len(iterations), label='Mean FPS')
        plt.title('FPS')
        plt.xlabel('Iteration')
        plt.ylabel('FPS')
        plt.legend()
        plt.savefig(os.path.join(plotDir, 'FpsPerIter.png'), bbox_inches='tight')

    # Pie plot for class distribution (the count for some classes is incorrect, just by 1)
    classNames = MetadataCatalog.get("my_dataset_val").thing_classes
    nClassInstances = np.zeros(len(classNames))
    for img in DatasetCatalog.get("my_dataset_val"):
        for ann in img['annotations']:
            if ann['iscrowd'] == 0:
                nClassInstances[ann['category_id']] += 1

    def func(pct, allvalues):
        #  Function to show percentage and absolute value in the pie plot
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    plt.clf()
    plt.figure(figsize=(15, 9))
    plt.pie(nClassInstances, labels=classNames, autopct=lambda pct: func(pct, nClassInstances))
    plt.title('Class distribution')
    plt.savefig(os.path.join(plotDir, 'classDistributionPiePlot.png'), bbox_inches='tight')

    # Inference -----
    inference = inference_on_dataset(predictor.model, val_loader, evaluator)
    with open(os.path.join(testDir, 'inferenceResults.json'), 'w') as f:
        json.dump(inference, f)

    cocoGt = COCO(MetadataCatalog.get("my_dataset_val").json_file)
    cocoPred = cocoGt.loadRes(os.path.join(testDir, 'coco_instances_results.json'))
    cocoEval = COCOeval(cocoGt, cocoPred, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # Graphs post-inference -----

    # Bar plot for AP per category
    x = np.arange(len(classNames))  # label locations
    width = 0.35  # bars width

    plt.clf()
    plt.figure(figsize=(15, 9))
    fig, ax = plt.subplots(figsize=(15,9))
    apPerClass = np.around((list(inference['bbox'].values()))[-len(classNames):], 2)
    idxOrder = np.argsort(apPerClass)
    rects1 = ax.bar(x - width / 2, apPerClass[idxOrder], width, label='AP')
    rects2 = ax.bar(x + width / 2, nClassInstances/10, width, label='Number of instances / 10')
    ax.bar_label(rects1, padding=4)
    ax.bar_label(rects2, padding=4)

    ax.set_title('Scores by group and gender')
    ax.set_xticks(x, np.array(classNames)[idxOrder])
    ax.set_ylabel('')
    ax.legend()
    fig.savefig(os.path.join(plotDir, 'apAndDistributionPerClassBarPlot.png'), bbox_inches='tight')

    # Horizontal bar plot for different AP metrics
    plt.clf()
    plt.figure(figsize=(15, 9))
    aps = np.around((list(inference['bbox'].values()))[:6], 2)
    apsLabels = list(inference['bbox'].keys())[:6]

    plt.barh(apsLabels, aps)
    plt.title('AP general metrics')
    plt.savefig(os.path.join(plotDir, 'apsHrzBarPlot.png'), bbox_inches='tight')

    # -----
    print(inference.keys())
    print(cocoEval.summarize())
    print(cocoEval.stats)
    print(inference['bbox'])
    print(cocoEval.params.recThrs)
    print(cocoEval.params.iouThrs)
