import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class Model(nn.Module):
    """
    Class to instantiate EfficientNet-b5 model object which only
    used images as inputs.
    """
    def __init__(self, model_name="efficientnet-b5", pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.model_name = model_name
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = getattr(self.backbone, "_fc").in_features
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.pool_type(self.backbone.extract_features(x), 1)
        features = features.view(x.size(0), -1)
        return self.classifier(features)
    
    
# class Model(nn.Module):
#     """
#     Class to instantiate EfficientNet-b5 model object which uses images
#     as well as tabular features as inputs.
#     """
#     def __init__(self, model_name='efficientnet-b5', pool_type=F.adaptive_avg_pool2d,
#                 num_tabular_features=0):
#         super().__init__()
#         self.pool_type = pool_type
#         self.model_name = model_name
#         self.backbone = EfficientNet.from_pretrained(model_name)
#         in_features = getattr(self.backbone, "_fc").in_features
#         if num_tabular_features>0:
#             self.meta = nn.Sequential(
#                 nn.Linear(num_tabular_features, 512),
#                 nn.BatchNorm1d(512),
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(512, 128),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU())
#             in_features += 128
#         self.output = nn.Linear(in_features, 1)
    
#     def forward(self, image, tabular_features=None):
#         features = self.pool_type(self.backbone.extract_features(image), 1)
#         cnn_features = features.view(image.size(0),-1)
#         if num_tabular_features>0:
#             tabular_features = self.meta(tabular_features)
#             all_features = torch.cat((cnn_features, tabular_features), dim=1)
#             output = self.output(all_features)
#             return output
#         else:
#             output = self.output(cnn_features)
#             return output
