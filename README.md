# Traffic Sign Detection Base on PyQt5 and OpenCV

Created by Haowen Wang (王浩闻), Yicong Wang (王逸聪) and Jian Zhao (赵健) from Shanghai Jiao Tong University.

## Introduction
**This is a course design project. *Info: Computer Graphics, ME6180-020-M01, Shanghai Jiao Tong University.***

Detect traffic sign using OpenCV and Python. Realize UI interface and human-computer interaction through PyQt5. Two versions are currently available: Single and Online Detection &amp; Environment Detection.

Our project implement and improve the following model architecture:
> Single Detection:  https://www.youtube.com/watch?v=SWaYRyi0TTs, https://github.com/Fafa-DL/Opencv-project
> 
> Environment Detection: https://github.com/394781865/Traffic_sign_detect

## Installation

Different version of detection has different implementation environment. We strongly recommend using **Anaconda** environment or **virtualenv** to create environment.

To create/activate/exit Anaconda environment:

        conda create -n {ENV_NAME}
        conda activate {ENV_NAME}
        conda deactivate

### 1. Single and Online Detection

The code has been tested with Python 3.9, Tensorflow 2.6.0, PyQt 5.15.6 on macOS Big Sur, Windows 10 and Windows 11.

### 2. Environment Detection
The code has been tested with Python 3.7, Tensorflow 1.15.0, PyQt 5.15.6 on macOS Big Sur, Windows 10 and Windows 11.

## Usage
### Use pre-trained model
To use pre-trained model and show PyQt5 interface:

        python ./SingleDetection/main.py
        python ./EnvDetection/main.py

### Train on your own data
#### Single and Online Detection Version
After you prepare the dataset, you can simply run:

        python ./SingleDetection/TSD_single_train.py

Note that you may need to modify the code appropriately to adapt to different data formats.
#### Environment Detection Version
After you prepare the dataset, you can run:

        python ./EnvDetection/example/train_P_net.py
        python ./EnvDetection/example/train_R_net.py
        python ./EnvDetection/example/train_O_net.py

After trained the cascaded network, you have to adapt the files in data_providers folder to fit your dataset, then run:

        python ./SingleDetection/TSD_env_train.py

## License

Our code is released under MIT License. See the LICENSE file for more details.