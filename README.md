# DAC4TB

## Training

### Installation
1. Install CUDA Toolkit `https://developer.nvidia.com/cuda-10.1-download-archive-update2`
2. Install Anaconda `https://www.anaconda.com/distribution/#download-section`

### Run Training
1. Open Jupyter Notebook from Anaconda Navigator
2. Open file `predict.ipynb`
3. Copy datasets into folder `Dataset\DATASET_NAME`<br>
   Seperate each images labels into sub folder:<br>
   `Dataset\DATASET_NAME\ABNORMAL`<br>
   `Dataset\DATASET_NAME\NORMAL`<br>
   Only *.jpg is allow.
4. Change dataset name in line `dataset_name = "DDC Prison BLM TUA"`
5. Change model name in line `MODEL_NAME = 'blm_2_t_mn'`<br>
  Training model will be save into folder `saved_model/`
6. Click run to training

## Predict

### Installation
1. Install docker `https://www.docker.com/products/docker-desktop`
2. Right click from docker icon in taskbar then click `Settings`
3. Click `Resources\FILE SHARING` then tick drive that you want to save input and output folder.<br>
  Default is drive `C:`.<br>
  Then click `Apply & Restart` Button
4. Increase docker cpus and memory resource from `Resources\ADVANCED` then click `Apply & Restart` Button
5. Create input and output folders in `C:\input` and `C:\output`

### Run Prediction
1. Copy images that want to labels into `c:\input\000xxxx.jpg`.<br>
  Only *.jpg is allow.
2. Open Command Prompt then run these command.<br>
  `docker run -v c:/input:/input -v c:/output:/output -it asia.gcr.io/thaihealthai/dac4tb:blm_2_t_mn-1.0.0 python predict.py`
3. The results will be in folder `c:\output\output_yyyymmmdddhhmmss.csv`
3. If input and output folders different from default. Please change `-v d:/input_folder:/input` `-v d:/output_folder/:output`.
4. Change other dac4tb version from `asia.gcr.io/thaihealthai/dac4tb:blm_2_t_mn-1.0.0`
