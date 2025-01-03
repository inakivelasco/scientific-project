from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
# from ..common.data.coco import dataloader
from ..common.data.maize import dataloader
from ..common.models.fcos import model
from ..common.train import train
# from detectron2.config import LazyCall as L
# solver = L()
dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.001  # Default 0.01

model.backbone.bottom_up.freeze_at = 2
model.num_classes = 3
dataloader.train.total_batch_size = 2

# train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # Use this one for the first train
train.init_checkpoint = "/Scientific-Project/models/fcos_R_50_FPN_1x_v00/model_final.pth"  # Get previous checkpoint