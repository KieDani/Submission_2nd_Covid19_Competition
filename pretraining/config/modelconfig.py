from pretraining.models.upernext.upernext import UperNeXt, MultiNeXt
from pretraining.models.unet import UnetSegmentation

class ConvNeXt3DConfig:
    def __init__(self):
        super(ConvNeXt3DConfig, self).__init__()
        self.added_dim = 2
        self.optim_name = "adamw"
        self.learning_rate = 3e-4
        self.weight_decay = 1e-8
        self.pretrained = True
        self.init_mode = 'two_g' #full, one, two, three, one_m, two_m, three_m, one_g, two_g, three_g
        self.ten_net = 0 #if 0 -> use ordinary linear layer
        self.in_chan = 3

class UnetConfig:
    def __init__(self):
        super(UnetConfig, self).__init__()
        self.optim_name = "adamw"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-8
        self.in_chan = 3
        self.num_at_once = 1

class UpernextConfig:
    def __init__(self):
        super(UpernextConfig, self).__init__()
        self.added_dim = 2
        self.optim_name = "adamw"
        self.learning_rate = 1e-4
        self.weight_decay = 1e-8
        self.pretrained = True
        self.init_mode = 'two_g'  # full, one, two, three, one_m, two_m, three_m, one_g, two_g, three_g
        self.ten_net = 0  # if 0 -> use ordinary linear layer
        self.in_chan = 1
        self.use_transformer = False
        self.num_at_once = 4
        self.size = 'tiny'
        self.drop_path = 0.1 if self.size == 'tiny' else 0.4

class MultinextConfig:
    def __init__(self):
        super(MultinextConfig, self).__init__()
        self.added_dim = 2
        self.optim_name = "adamw"
        self.learning_rate = 1e-5
        self.weight_decay = 1e-8
        self.pretrained = True
        self.pretrained_mode = 'imagenet'
        self.init_mode = 'two_g'  # full, one, two, three, one_m, two_m, three_m, one_g, two_g, three_g
        self.ten_net = 0  # if 0 -> use ordinary linear layer
        self.in_chan = 1
        self.use_transformer = False
        self.num_at_once = 4
        self.size = 'tiny'
        self.drop_path = 0.1 if self.size == 'tiny' or self.size == 'micro' else 0.4
        self.lr_decay = None
        self.cosine_decay = None
        self.use_metadata = True




def get_model(config):
    modelconfig = config.modelconfig
    model_name = config.MODEL_NAME
    if model_name == 'upernext':
        model = UperNeXt(modelconfig, config)
    elif model_name == 'multinext':
        model = MultiNeXt(modelconfig, config)
    elif model_name == 'unet':
        model = UnetSegmentation(modelconfig)
    else:
        raise KeyError(f"Unknown model: {model_name}")

    return model

def get_modelconfig(config):
    model_name = config.MODEL_NAME
    if model_name == 'upernext':
        modelconfig = UpernextConfig()
    elif model_name == 'multinext':
        modelconfig = MultinextConfig()
    elif model_name == 'unet':
        modelconfig = UnetConfig()
    else:
        raise KeyError(f"Unknown model: {model_name}")

    return modelconfig