# Scientific Project


This project is based in the detectron2 repository (https://github.com/facebookresearch/detectron2). It consists of a general pipeline with the following steps:

![Pipeline](/docs/pipeline.png)



## Generated report
A report will be generated at the end of the pipeline execution. It will include the following aspects:
- Dataset description
- Optional extra information
- Model training plots
- Test plots using the provided dataset
- Images & their predictions

Some report examples are available in the [reports folder](/reports).


## Previous considerations
### Images and annotations format
This project currently works with COCO format annotations.
### Dataset structure
The actual code requires a specific structure for the folder containing the evaluating dataset:

![Dataset structure](/docs/datasetStructure.jpg)

As it can be seen in the previous image, the dataset folder name will correspond to the dataset name. In addition, a description .txt file, an annotations .json file and an images folder should be inside the dataset folder.


## Usage instructions
### Stage 1: Training
Previous possible changes:
- [train.py](detectron2-code/configs/common/train.py): adjust parameters such as training iterations
- [fcos_R_50_FPN_1x_maize.py](detectron2-code/configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py): adjust parameters such as initial checkpoint
- [lazyconfig_train_net.py](detectron2-code/tools/lazyconfig_train_net.py): adjust parameters such as images/annotations paths (in **register_dataset()**) or output folder

`lazyconfig_train_net.py --config-file ../configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py --num-gpus 1 --num-machines 1`
### Stage 2: Evaluation
### Stage 3: Plot generation
### Stage 4: Report generation
