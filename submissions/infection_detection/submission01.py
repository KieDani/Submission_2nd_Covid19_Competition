### Submission 01: imagenet (full) pretraining with cross validation; rotOrProb = 0.0
from training.config.config import BaseConfig
from training.training import run
from training.misc_utilities.determinism import set_deterministic
import logging
import torch
import os
from collections import namedtuple
from pathlib import Path

import training.inference as inf
from paths import DATA_PATH, inference_checkpoint_dir
inference_checkpoint = os.path.join(inference_checkpoint_dir, "det1.pt")



def run_Inf(decayUntil=10, loss='balce', pretraining='imagenet', cv_enabeled=False, prenick='', init_mode = 'two_g', rotationProb = 0, augmentations = 'all', max_epochs=50):
    logging.getLogger().setLevel(logging.INFO)

    split = 'cv5_cov_nocov.csv' if cv_enabeled else 'official.csv'
    nickname = ''.join((
        prenick, '_', loss, '-', pretraining, "_lrDecayUntil", str(decayUntil), '_initMode', init_mode,
        '_rotProb', str(rotationProb), '_AugMode', augmentations
    ))
    # config = BaseConfig(split="cv5_infonly.csv", cv_enabled=True, num_steps=1, data_name="mia", nickname=loss + '-' + pretraining + "_lrDecayUntil" + str(decayUntil))
    config = BaseConfig(split=split, cv_enabled=cv_enabeled, num_steps=1, data_name="mia",
                        evaluate_official_val=True,
                        nickname=nickname)

    if loss == 'ce':
        config.modelconfig.loss_fn = "ce_inf_eccv"
    elif loss == 'balce':
        config.modelconfig.loss_fn = "balce_inf_eccv"
    else:
        print('wrong loss name')
    config.modelconfig.pretrained_mode = pretraining
    config.modelconfig.init_mode = init_mode
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = max_epochs
    config.MAX_EPOCHS_LRSCHEDULER = 30
    config.modelconfig.num_classes = 2
    config.modelconfig.decay_lr_until = decayUntil
    for phase in ['train', 'val', 'test']:
        config.dataconfigs[phase].do_normalization = True
        config.dataconfigs[phase].orientation_prob = rotationProb
        if augmentations == 'noDeform':
            config.dataconfigs[phase].deform_prob = 0
        if augmentations == 'none':
            config.dataconfigs[phase].flip = False
            config.dataconfigs[phase].rotate_prob = 0
            config.dataconfigs[phase].blur_prob = 0
            config.dataconfigs[phase].noise_prob = 0
            config.dataconfigs[phase].deform_prob = 0

    config.modelconfig.size = 'tiny'

    #config.LOGS_PATH = os.path.join(config.LOGS_PATH, prenick) if prenick != '' else config.LOGS_PATH

    set_deterministic()

    run(config, reset_deterministic=True)



def train():
    loss = 'balce'
    decayUntil = 0
    init_mode = 'full'
    augmentations = 'all'
    max_epochs = 70

    prenick = 'InfectionDetection/Submission01'
    cv_enabeled = True
    rotationProb = 0.
    pretraining = 'imagenet'

    run_Inf(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode, rotationProb, augmentations, max_epochs)


def pretrain():
    pass


def inference():
    args = {
        'gpu': '0',
        'checkpoint': inference_checkpoint,
        'input': os.path.join(DATA_PATH, 'test_cov19d', 'detection'),
        'output': 'output_det_submission01.csv',
    }
    args = namedtuple("args", args.keys())(*args.values())
    logging.getLogger().setLevel(logging.INFO)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    checkpoint = torch.load(args.checkpoint)
    searchdir = Path(args.input)
    inputs = searchdir.glob("ct_scan*")

    decayUntil = 0
    init_mode = 'full'
    augmentations = 'all'
    max_epochs = 70

    cv_enabeled = True
    rotationProb = 0.
    pretraining = 'imagenet'

    split = 'cv5_cov_nocov.csv' if cv_enabeled else 'official.csv'
    config = BaseConfig(split=split, cv_enabled=cv_enabeled, num_steps=1, data_name="mia",
                        evaluate_official_val=True,
                        nickname='prediction')

    config.modelconfig.loss_fn = "balce_inf_eccv"

    config.modelconfig.pretrained_mode = pretraining
    config.modelconfig.init_mode = init_mode
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min": 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = max_epochs
    config.MAX_EPOCHS_LRSCHEDULER = 50
    config.modelconfig.num_classes = 2
    config.modelconfig.decay_lr_until = decayUntil
    for phase in ['train', 'val', 'test']:
        config.dataconfigs[phase].do_normalization = True
        config.dataconfigs[phase].orientation_prob = rotationProb
        if augmentations == 'noDeform':
            config.dataconfigs[phase].deform_prob = 0
        if augmentations == 'none':
            config.dataconfigs[phase].flip = False
            config.dataconfigs[phase].rotate_prob = 0
            config.dataconfigs[phase].blur_prob = 0
            config.dataconfigs[phase].noise_prob = 0
            config.dataconfigs[phase].deform_prob = 0

    config.modelconfig.size = 'tiny'

    inf.inference(checkpoint, inputs, args.output, 'model_ema', config=config)



def main():
    #Look at readme to see how to start imagenet pretraining. Put the weights into the correct path!
    #pretrain()
    # train()
    inference()



if __name__ == "__main__":
    main()