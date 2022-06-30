import torch
from training.models.hypernet import HyperConvNeXt3dSTOIC
from training.models.slice2d import Slice2DModel
from training.models.resnet import ResNet
from training.models.convnext import ConvNeXt3dSTOIC
from training.models.testmodel import Testmodel
from training.models.uniformer import UniFormer3DSTOIC

class ResNetConfig:
    """
    class for configuration files of model hyperparameters
    """

    def __init__(self):
        self.loss_fn = "bce_inf_sev"
        self.pretrained = True
        self.optim_name = "adam"
        self.learning_rate = 1e-5
        self.weight_decay = 1e-8
        self.pos_weight = 3.0  # None

class ConvNeXt3DConfig:
    def __init__(self, num_trained_stages='4'):
        self.loss_fn = "ce_inf_eccv"
        self.num_classes =  4
        self.added_dim = 2
        self.optim_name = "adamw"
        #self.learning_rate = 1e-5
        self.learning_rate = 5e-5
        self.weight_decay = 1e-8
        self.pretrained = True
        # self.pretrained_mode = 'imagenet'
        self.pretrained_mode = 'TciaMosmed'
        #self.pretrained_mode = 'multitask' #You should set do_normalization=True in dataconfig
        self.init_mode = 'two_g' #full, one, two, three, one_m, two_m, three_m, one_g, two_g, three_g
        self.ten_net = 0 #if 0 -> use ordinary linear layer
        self.in_chan = 1
        self.use_transformer = False
        self.use_metadata = True
        self.size = 'tiny'
        # self.size = 'small'
        self.train_stages = [i for i in range(4-int(num_trained_stages), 4)]
        self.siamese = False
        #self.pos_weight = None
        self.pos_weight = 1.0
        self.drop_path = 0.0
        #self.lr_decay = {
        #    "gamma" : 0.5,
        #    "every_num_epochs" : 3
        #}
        self.lr_decay = None
        # self.cosine_decay = None
        self.cosine_decay = {
            "lr_min" : 1e-6
        }
        self.decay_lr_until = 20 # Maximum epoch until which to use lr decay

class HyperConvNeXt3DConfig:
    def __init__(self):
        self.loss_fn = "bce_inf_sev"
        self.pretrained = True
        self.added_dim = 2
        self.optim_name = "adamw"
        self.learning_rate = 1e-5
        self.weight_decay = 0.0 #1e-8
        self.pos_weight = 3.0  # None

class UniFormer3DConfig:
    def __init__(self):
        self.loss_fn = "bce_inf_sev"
        self.pretrained = True
        self.added_dim = 2
        self.optim_name = "adamw"
        self.learning_rate = 1e-5
        self.weight_decay = 1e-8
        self.pretrained = True
        self.init_mode = 'full' #full, one, two, three, one_m, two_m, three_m, one_g, two_g, three_g
        self.pos_weight = 3.0  # None


class TestmodelConfig:
    def __init__(self):
        self.loss_fn = "bce_inf_sev"
        self.dropout = 0.3
        # TODO: I have no idea what this model needs as optim and lr so copied from config.py
        self.optim_name = "adam"
        self.learning_rate = 1e-5
        self.pos_weight = 3.0  # None

class Slice2DModelConfig:
    def __init__(self):
        self.backbone = Slice2DModel.BB_CONVNEXT_T
        self.dropout = 0.3
        self.use_age = True
        self.use_sex = True
        self.hidden_features = 128
        self.transform_meta = 0
        # must always be True, regardless if use_age and use_sex are both False
        # if set to True, the model wil recive three inputs instead of one
        self.use_metadata = True
        self.loss_fn = "3ce"

        # TODO: I have no idea what this model needs as optim and lr so copied from config.py
        self.optim_name = "adam"
        self.learning_rate = 1e-5
        self.weight_decay = 0
        self.sex_classes = 2
        self.pos_weight = 3.0  # None


def get_modelconfig(model_name, num_trained_stages='4'):
    if model_name == 'resnet':
        return ResNetConfig()
    elif model_name == "convnext":
        return ConvNeXt3DConfig(num_trained_stages)
    elif model_name == "hyperconvnext":
        return HyperConvNeXt3DConfig()
    elif model_name == "uniformer":
        return UniFormer3DConfig()
    elif model_name == 'testmodel':
        return TestmodelConfig()
    elif model_name == "slice2d":
        return Slice2DModelConfig()

    raise KeyError(f"Unknown model: {model_name}")


def get_model(config):
    modelconfig = config.modelconfig
    if isinstance(modelconfig, ResNetConfig):
        return ResNet(modelconfig)
    if isinstance(modelconfig, ConvNeXt3DConfig):
        return ConvNeXt3dSTOIC(modelconfig, config)
    if isinstance(modelconfig, HyperConvNeXt3DConfig):
        return HyperConvNeXt3dSTOIC(modelconfig)
    if isinstance(modelconfig, UniFormer3DConfig):
        return UniFormer3DSTOIC(modelconfig)
    if isinstance(modelconfig, TestmodelConfig):
        return Testmodel(modelconfig)
    if isinstance(modelconfig, Slice2DModelConfig):
        return Slice2DModel(modelconfig)

    raise KeyError(f"Unknown model: {modelconfig}")


def get_optimizer(modelconfig, model):
    if modelconfig.optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=modelconfig.learning_rate, weight_decay=modelconfig.weight_decay)
    if modelconfig.optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=modelconfig.learning_rate, weight_decay=modelconfig.weight_decay)

    raise KeyError(f"Unknown optimizer: {modelconfig.optim_name}")
