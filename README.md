# Traffic Sign Detection Based on PyQt5 and OpenCV

Created by Haowen Wang (王浩闻), Yicong Wang (王逸聪) and Jian Zhao (赵健) from Shanghai Jiao Tong University.

![Version 1: Single and Online Detection](https://github.com/wang-h-w/traffic-sign-detection/blob/master/Single_and_Online_Detection.png)

![Version2: Environment Detection](https://github.com/wang-h-w/traffic-sign-detection/blob/master/Environment_Detection.png)

## Introduction
**This is a course design project. *Info: Computer Graphics, ME6180-020-M01, Shanghai Jiao Tong University.***

Detect traffic sign using OpenCV and Python. Realize UI interface and human-computer interaction through PyQt5. Two versions are currently available: Single and Online Detection &amp; Environment Detection.

Demo videos can be downloaded from this link: https://jbox.sjtu.edu.cn/l/v1Xvbl

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

### 0. Install Requirements
The requirements of SingleDetection and EnvDetection is different. Use following command to install:

        pip install -r ./SingleDetection/requirements_single.txt
        pip install -r ./EnvDetection/requirements_env.txt

### 1. Single and Online Detection
The code has been tested with Python 3.9, Tensorflow 2.6.0, PyQt 5.15.6 on macOS Big Sur, Windows 10 and Windows 11.

### 2. Environment Detection
The code has been tested with Python 3.7, Tensorflow 1.15.0, PyQt 5.15.6 on macOS Big Sur, Windows 10 and Windows 11.

## Usage
### 1. Use pre-trained model
To use pre-trained model and show PyQt5 interface:

        python ./SingleDetection/main.py
        python ./EnvDetection/main.py

### 2. Train on your own data
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

### 3. Run on your Windows! (NEW!)
No need to setup the environment. We now provide an .exe application that can be run directly on Windows 10/11. Only for **SingleDetection**.

You can download our application from: https://jbox.sjtu.edu.cn/l/xFFNV8. Just click **main.exe** and detect the sign!

## License

Our code is released under MIT License. See the LICENSE file for more details.