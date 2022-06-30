from training.config.config import BaseConfig
from training.training import run
from training.misc_utilities.determinism import set_deterministic
import os
import logging
from training.config.dataconfig import get_dataset
from tqdm import tqdm
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train ECCV-AIMA network')
    # general

    parser.add_argument('--gpu',
                        default="6",
                        help='gpu id',
                        type=str)

    args = parser.parse_args()
    return args

def run_Sev(decayUntil = 20, loss='balce', pretraining='imagenet', cv_enabeled = False, prenick='', init_mode = 'two_g', rotationProb = 0, augmentations = 'all', max_epochs=60):
    logging.getLogger().setLevel(logging.INFO)

    split = 'cv5_infonly.csv' if cv_enabeled else 'eccv.csv'
    #config = BaseConfig(split="cv5_infonly.csv", cv_enabled=True, num_steps=1, data_name="mia", nickname=loss + '-' + pretraining + "_lrDecayUntil" + str(decayUntil))
    nickname = ''.join((
        prenick, '_', loss, '-', pretraining, "_lrDecayUntil", str(decayUntil), '_initMode', init_mode,
        '_rotProb', str(rotationProb), '_AugMode', augmentations
    ))
    config = BaseConfig(split=split, cv_enabled=cv_enabeled, num_steps=1, data_name="mia",
                        evaluate_official_val=True,
                        nickname=nickname)

    if loss == 'ce':
        config.modelconfig.loss_fn = "ce_sev_eccv"
    elif loss == 'wce':
        config.modelconfig.loss_fn = "wce_sev_eccv"
    elif loss == 'l2':
        config.modelconfig.loss_fn = "l2_sev_eccv"
    elif loss == 'balce':
        config.modelconfig.loss_fn = "balce_sev_eccv"
    else:
        print('wrong loss name')
    config.modelconfig.pretrained_mode = pretraining
    config.modelconfig.init_mode = init_mode
    config.modelconfig.pos_weight = 1.0
    # Parameters from the ConvNeXt paper
    config.modelconfig.learning_rate = 5e-5
    config.modelconfig.cosine_decay = {
        "lr_min" : 1e-6
    }
    config.Batch_SIZE = 4
    config.MAX_EPOCHS = max_epochs
    config.MAX_EPOCHS_LRSCHEDULER = 50
    config.modelconfig.num_classes = 4
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


def compare_pretrainings():
    loss = 'balce'
    decayUntil = 10
    cv_enabeled = False
    prenick = 'comparePretrainings'
    prenick = os.path.join(prenick, prenick)

    for pretraining in ['imagenet', 'segmiaECCV', 'multimiaECCV', 'multitaskECCV', 'segmentationECCV', 'segmentation', 'trainedMiaInf', 'trainedMiaInfECCV']:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick)


def compare_imagenetInitialization():
    loss = 'balce'
    decayUntil = 10
    pretraining = 'imagenet'
    cv_enabeled = False
    prenick = 'compareImagenetInitialization'
    prenick = os.path.join(prenick, prenick)

    for init_mode in ['one_g', 'full', 'two_g']:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode)


def compare_lossfn():
    decayUntil = 10
    pretraining = 'segmentationECCV'
    cv_enabeled = False
    prenick = 'compareLossfn'
    prenick = os.path.join(prenick, prenick)

    for loss in ['balce', 'ce', 'wce']:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick)


def compare_scheduler():
    loss = 'balce'
    pretraining = 'segmentationECCV'
    cv_enabeled = False
    prenick = 'compareScheduler'
    prenick = os.path.join(prenick, prenick)

    #decayUntil = 0 means that no learning rate scheduler is used
    for decayUntil in [0, 5, 10, 20, 30]:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick)

    prenick = 'compareSchedulerInf'
    for decayUntil in [0, 5, 10, 20, 30]:
        run_Inf(decayUntil, loss, pretraining, cv_enabeled, prenick)


def compare_rotations():
    loss = 'balce'
    decayUntil = 10
    pretraining = 'segmentationECCV'
    cv_enabeled = False
    prenick = 'compareRotations'
    prenick = os.path.join(prenick, prenick)

    for rotationProb in [0, 0.1, 0.25]:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick, rotationProb=rotationProb)


def compare_augmentations():
    loss = 'balce'
    decayUntil = 10
    pretraining = 'segmentationECCV'
    cv_enabeled = False
    prenick = 'compareAugmentations'
    prenick = os.path.join(prenick, prenick)

    for aug in ['all', 'noDeform', 'none']:
        run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick, augmentations=aug)


def cv_Severity():
    loss = 'balce'
    decayUntil = 10
    pretraining = 'segmentationECCV'
    cv_enabeled = True
    prenick = 'SeverityCV'
    prenick = os.path.join(prenick, prenick)
    init_mode = 'two_g'
    rotationProb = 0
    augmentations = 'all'

    run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode=init_mode, rotationProb=rotationProb, augmentations=augmentations)


def cv_Infection():
    loss = 'balce'
    decayUntil = 10
    pretraining = 'segmentationECCV'
    cv_enabeled = True
    prenick = 'InfectionCV'
    prenick = os.path.join(prenick, prenick)
    init_mode = 'two_g'
    rotationProb = 0
    augmentations = 'all'

    run_Inf(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode=init_mode, rotationProb=rotationProb, augmentations=augmentations)


def final_Severity():
    loss = 'balce'
    decayUntil = 0
    init_mode = 'full'
    augmentations = 'all'
    max_epochs = 100

    for prenick, cv_enabeled in [('FinalSeverityNoCV', False), ('FinalSeverityCV', True)]:
        for rotationProb in [0., 0.25]:
            for pretraining in ['imagenet', 'segmentationECCV', 'segmiaECCV', 'multitaskECCV', 'trainedMiaInfECCV', 'segmentationECCVFull', 'segmiaECCVFull', 'multitaskECCVFull']:
                run_Sev(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode, rotationProb, augmentations,
                        max_epochs)


def final_Infection():
    loss = 'balce'
    decayUntil = 0
    init_mode = 'full'
    augmentations = 'all'
    max_epochs = 70

    for prenick, cv_enabeled in [('FinalInfectionNoCV', False), ('FinalInfectionCV', True)]:
        for rotationProb in [0., 0.25]:
            for pretraining in ['imagenet', 'segmiaECCV', 'multitaskECCV']:
                run_Inf(decayUntil, loss, pretraining, cv_enabeled, prenick, init_mode, rotationProb, augmentations,
                        max_epochs)





if __name__ == "__main__":
    #time.sleep(10800)
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # compare_imagenetInitialization()
    # compare_pretrainings()
    # compare_lossfn()
    # compare_scheduler()
    # compare_augmentations()
    # compare_rotations()

    # cv_Severity()
    # cv_Infection()

    #trainedMiaInfECCV Pretraining
    # run_Inf(decayUntil=10, loss='balce', pretraining='imagenet', cv_enabeled=False, prenick='pretraining/pretrainingTrainedMiaInfECCV', init_mode='full', rotationProb=0, augmentations='all')

    # final_Infection()
    # final_Severity()
