# Scientific Project


This project is based in the detectron2 repository (https://github.com/facebookresearch/detectron2). It consists of a general pipeline with the following steps:

![Pipeline](/docs/pipeline.png)
It is aimed to work with real and synthetic datasets.


## Generated report
A report will be generated at the end of the pipeline execution. It will include the following aspects:
- Dataset description
- Optional extra information
- Model training plots
- Test plots using the provided dataset
- Images & their predictions

Some report examples are available in the [reports folder](/reports).


## Previous considerations
***Some scripts could stop working if these considerations are not taken into account.***
### Windows 11
The whole project has been developed in Windows 11.
### Project structure
The directories structure of the project is already defined and it is the same as this repository. Nevertheless, the detectron2 repository should be downloaded first and the corresponding files must be substituted with the ones of this project.
### Model specification
This project currently works with **fcos_R_50_FPN_1x** object detection model.
### Dataset specification
This project currently works with an specific kind of dataset containing images for three different classes: **bark**, **maize** and **weeds**.
### Images and annotations format
This project currently works with COCO format annotations.
### Dataset structure
The actual code requires a specific structure for the folder containing the evaluating dataset:

![Dataset structure](/docs/datasetStructure.jpg)

As it can be seen in the previous image, the dataset folder name will correspond to the dataset name. In addition, a description .txt file, an annotations .json file and an images folder should be inside the dataset folder.


## Usage instructions
### Stage 1: Training [[lazyconfig_train_net.py](detectron2-code/tools/lazyconfig_train_net.py)]
Previous changes:
- [train.py](detectron2-code/configs/common/train.py): adjust parameters such as training iterations
- [fcos_R_50_FPN_1x_maize.py](detectron2-code/configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py): adjust parameters such as initial checkpoint
- [lazyconfig_train_net.py](detectron2-code/tools/lazyconfig_train_net.py): adjust parameters such as images/annotations paths (in **register_dataset()**) or output folder
- Adjust args parameters

`lazyconfig_train_net.py --config-file ../configs/COCO-Detection/fcos_R_50_FPN_1x_maize.py --num-gpus 1 --num-machines 1`
### Stage 2: Evaluation [[evaluate_lazy_config.py](detectron2-code/tools/evaluate_lazy_config.py)]
Previous changes:
- [evaluate_lazy_config.py](detectron2-code/tools/evaluate_lazy_config.py): adjust **register_dataset()** so that the train dataset is correct and the validation dataset is the dataset the user wants to evaluate. Adjust **cfg.train.output_dir** to match the corresponding output folder
- Adjust args parameters

`evaluate_lazy_config.py --config-file ../../models/fcos_R_50_FPN_1x_v01/config.yaml --num-gpus 1`
### Stage 3: Plot generation [[generatePlots.py](ownCode/generatePlots.py)]
Previous changes:
- Adjust args parameters: --model-name and --dataset-name

`generatePlots.py --model-name fcos_R_50_FPN_1x_v01 --dataset-name test`
### Stage 4: Report generation [[generateReport.py](ownCode/generateReport.py)]
Previous changes:
- Adjust args parameters: --model-name and --dataset-dir

`generateReport.py --model-name fcos_R_50_FPN_1x_v01 --dataset-dir some/folders/test`
### Extra:
The model can be exported with the following command [[export_lazy_config.py](detectron2-code/tools/deploy/export_lazy_config.py)]:

`export_lazy_config.py --config-file ../../../models/fcos_R_50_FPN_1x_v01/config.yaml --format torchscript --output /Scientific-Project/export-output`




