import torch
import torch.nn as nn
import torchvision

from training.models.convnext import convnext_small, convnext_tiny

class Slice2DModel(nn.Module):
    BB_RESNET = "resnet18"
    BB_CONVNEXT_S = "convnext_s"
    BB_CONVNEXT_T = "convnext_t"

    def __init__(self, modelconfig):
        super().__init__()
        self.use_age = modelconfig.use_age
        self.use_sex = modelconfig.use_sex
        if modelconfig.backbone == Slice2DModel.BB_RESNET:
            self.backbone = torchvision.models.resnet18(pretrained=True)
            backbone_outputs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif modelconfig.backbone == Slice2DModel.BB_CONVNEXT_S:
            self.backbone = convnext_small(pretrained=True)
            backbone_outputs = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif modelconfig.backbone == Slice2DModel.BB_CONVNEXT_T:
            self.backbone = convnext_tiny(pretrained=True)
            backbone_outputs = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {modelconfig.backbone}")

        transform_meta = modelconfig.transform_meta
        if not modelconfig.use_age and not modelconfig.use_sex:
            transform_meta = 0

        if transform_meta == 0:
            self.meta_layer = nn.Identity()
            meta_layer_out = (1 * modelconfig.use_age) + (modelconfig.sex_classes * modelconfig.use_sex)
        else:
            self.meta_layer = nn.Sequential(
                nn.Linear((1 * modelconfig.use_age) + (modelconfig.sex_classes * modelconfig.use_sex), transform_meta),
                nn.ReLU(),
            )
            meta_layer_out = transform_meta

        self.head = nn.Sequential(
            nn.Linear(backbone_outputs + meta_layer_out, modelconfig.hidden_features),
            nn.ReLU(),
            nn.Dropout(p=modelconfig.dropout),
            nn.Linear(modelconfig.hidden_features, 3),
        )

        self.last_device = None
    
    def forward(self, x: torch.Tensor, age: torch.Tensor, sex: torch.Tensor):
        x = self.backbone(x)

        if self.use_age or self.use_sex:
            to_cat = []
            if self.use_age:
                to_cat.append(age.unsqueeze(1))
            if self.use_sex:
                to_cat.append(sex)
            assert len(to_cat) > 0
            meta_x = torch.cat(to_cat, dim=1).float()
            try:
                meta_x = self.meta_layer(meta_x)
            except:
                print(age.dtype, sex.dtype)
                raise

            x_cat = torch.cat([x, meta_x], dim=1)
        else:
            x_cat = x

        try:
            xh = self.head(x_cat)
        except:
            if self.use_age or self.use_sex:
                print("darn it", x.shape, meta_x.shape, self.use_age, self.use_sex)
            else:
                print("darn it", x.shape, self.use_age, self.use_sex)
            raise
        return xh

    def to(self, device):
        # TODO: check if this is really required
        self.backbone.to(device)
        self.head.to(device)
        self.meta_layer.to(device)

        return super().to(device)
