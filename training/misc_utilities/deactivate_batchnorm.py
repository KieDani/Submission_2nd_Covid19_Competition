import torch

def set_all_bn_eval(model):
    mod_list = list(model.modules())

    for mod in mod_list:
        isbn = isinstance(mod, torch.nn.BatchNorm3d) or \
            isinstance(mod, torch.nn.BatchNorm2d) or \
            isinstance(mod, torch.nn.BatchNorm1d)
        if isbn:
            mod.eval()

    return model 