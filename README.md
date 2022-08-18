# COVID Detection and Severity Prediction with 3D-ConvNeXt and Custom Pretrainings

This is the code for our submissions at the 2nd Covid19 Competition.
We provide trained models for inference. 
Furthermore, we also provide the pretrained models in order to reproduce our trainings.
Soon, the instructions for the execution of the pretraining code will be added


## Python environment:

    python3 -m venv env
    . ./env/bin/activate
    pip install --upgrade pip
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip install wheel
    pip install -r min_requirements.txt



## Data preparation:
Set DATA_PATH = 'Path/TO/DATASET' in paths.py. The dataset will be further processed and saved in cache_path in paths.py. It is recommended to use absolute paths. Store the cov19d-dataset in 'Path/TO/DATASET'.
The data should be structured as:
Path/TO/DATASET/
  - train_cov19d
    - covid
      - ct_scan_xxx
      - ...
    - non-covid
      - ct_scan_xxx
      - ...
    - train_partition_covid_categories.csv

  - validation_cov19d
    - covid
      - ct_scan_xxx
      - ...
    - non-covid
      - ct_scan_xxx
      - ...
    - val_partition_covid_categories.csv

  - test_cov19d
    - detection
      - ct_scan_xxx
      - ...
    - severity
      - part_2_test_set_ECCV_22/
        - test_ct_scan_xxx
        - ...

Execute the skript cov19d.py with python -m training.data.cov19d in order to create the labels-file my_reference.csv

You have to define the paths in paths.py


## Inference:
Adjust inference_checkpoint = 'PATH/TO/CHECKPOINTS' in paths.py
Download the trained models using the `download_checkpoints.sh` script. You will get a file named `Checkpoints.zip`. Extract the checkpoints archive and put all checkpoint files (they end with `.pt`) inside the folder `checkpoints`. You should now have a file named `checkpoints/sev1.pt`, for example.

Run the submission scripts to reproduce our submission results, for example:

    # submission 1 for infection detection
    python -m submissions.infection_detection.submission01
    # submission 3 for severity prediction
    python -m submissions.severity_prediction.submission03

You will get files named like output_sev_submission03.csv. To convert these files to the challenge format with separate files for each class, use the `finalize_csvs.py` script.


## Training:
Set `PRETRAINED_PATH = '/PATH/TO/PRETRAINED/MODELS/'` in paths.py
Download the pretrained models using the `download_saved_models.sh` script. You will get a file named `saved_models.zip`.  Put the downloaded directory 'saved_models' into './'.
Uncomment the train()-function inside the main()-function of submissionXX.py
For the infection detection challenge, execute submission XX with python -m submissions.infection_detection.submissionXX
For the severity prediction challenge, execute submission XX with python -m submissions.severity_prediction.submissionXX

TODO: explain how to wrapup cross-validation results!!! CV-Modell erzeugen: `python3 -m misc_utilities.wrapup_checkpoint`
If you would like to use a custom checkpoint from a previous training, use `python training/misc_utilities/wrapup_checkpoints.py` to merge selected cross validation checkpoints for inference.


## Imagenet pretraining:
You can run the imagenet-pretraining with python -m training.pretraining.main --model convnext_pretraining --drop_path 0.1 --batch_size 256 --lr 4e-3 --update_freq 16 --model_ema true --model_ema_eval true --data_path YOUR/DATA/PATH --output_dir YOUR/OUTPUT/DIR --log_dir YOUR/LOG/DIR


# Pretrainings:
THE EXPLANATION WILL BE ADDED SOON.









