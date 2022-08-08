# Udacity AWS Sagemaker Capstone
udacity AWS Sagemaker Tuutorial Capstone project



### 1. Environment Setup

- Install [conda environment](https://docs.conda.io/projects/conda/en/latest/glossary.html#silent-mode-glossary)
    ```bash
    conda env remove -n cap
    conda create --name cap python=3.7
    conda activate cap 
    ```
- Install [YoloX](https://github.com/Megvii-BaseDetection/YOLOX/)
    -   ```bash
        git clone https://github.com/Megvii-BaseDetection/YOLOX
        export YOLOX_HOME=/home/rahul/workspace/coursera-sdsc-words/udacity-aws-sagemaker-capstone/YOLOX    
        
        ```
    -   ```bash
        cd $YOLOX_HOME 
        pip install -r requirements.txt
        pip install -v -e . 
        wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
        ```
    -   ```bash 
        pip install wandb
        pip install tensorboard
        wandb login
        ```
- Demo Test for Image - [Documentation](https://yolox.readthedocs.io/en/latest/quick_run.html)
    -   ```bash
        python tools/demo.py image -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/biker1.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```
    -   ```bash
        python tools/demo.py video -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/cyclists.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```
- Dataset [MSCOCO for Detection 2017](https://cocodataset.org/#download)
    - Download `Detection 2017` Train/Test/Val annotation and images dataset
    -   ```bash
        COCO/
            annotations/*.json           # contains annotation label as instances_{train,val}2017.json
            {train,val}2017/*.jpg        # contains image files that are mentioned in the corresponding json
        ```
    -   ```bash
        export COCO_HOME=/media/rahul/HIT-GRAID/data/coco
        ```
- Train setup test
    -   ```bash
        cd $YOLOX_HOME
        ln -s $COCO_HOME ./datasets/COCO
        python -m yolox.tools.train -f ./exps/default/yolox_nano.py -d 1 -b 8 --fp16  --logger wandb wandb-project yolox
        ```

pip uninstall setuptools
pip install setuptools==59.5.0

###  Tasks

- [ ] Setup Dataset and sample 100 images for each sample
- [ ] Training Loader updates
- [ ] Train the model 
- [ ] Test & Evaluate model
- [ ] Write report