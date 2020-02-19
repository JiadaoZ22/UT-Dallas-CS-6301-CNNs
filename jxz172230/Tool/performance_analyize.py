import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from pytorch_modelsize import SizeEstimator

# vgg19 = models.vgg19()
resNet50 = models.resnet50()
# se = SizeEstimation(model, input_size=(1,1,32,32))
estimate = se.estimate_size()
summary(resNet50, (3, 224, 224))