#Here, the preprocessed data is saved
cache_path = 'PATH/TO/CACHE/DIRECTORY'
#The path to the challenge dataset
DATA_PATH = 'PATH/TO/DATA/DIRECTORY'
#path to the logs directory
LOGS_PATH = 'PATH/TO/LOGS/DIRECTORY'
#path to the pretrained model directory
PRETRAINED_PATH = './saved_models/'


#path to pretraining datafiles -> create resized files acording to pretraining.data.data_utils
pretraining_DATA_PATH = 'PATH/TO/PRETRAINING/DATA/DIRECTORY' + '/resized256'
pretraining_LOGS_PATH = 'PATH/TO/LOGS/DIRECTORY'
pretraining_PRETRAINED_PATH = 'PATH/TO/PRETRAINED/MODELS/' + 'saved_models/'
pretraining_PATH_SEGMENTATION = 'PATH/TO/SUPERVISED/SEGMENTATION/DATASET'
pretraining_PATH_FULLTCIA = 'PATH/TO/UNLABELED/TCIA/DATASET'
pretraining_PATH_STOIC = 'PATH/TO/STOIC/DATASET'



#paths for inference
inference_checkpoint_dir = './checkpoints'