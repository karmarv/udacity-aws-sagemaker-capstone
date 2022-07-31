# udacity-aws-sagemaker-capstone
udacity AWS Sagemaker Tuutorial Capstone project



### 1. Environment Setup

- Install [conda environment](https://docs.conda.io/projects/conda/en/latest/glossary.html#silent-mode-glossary)
    ```bash
    conda create --name cap python=3.8
    conda activate cap 
    ```
- Install [YoloX](https://github.com/Megvii-BaseDetection/YOLOX/)
    ```bash
    cd YOLOX && pip install -r requirements.txt
    pip install -v -e . 
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
    ```
- Demo Test for Image
    ```bash
    python tools/demo.py image -n yolox-nano -c ./yolox_nano.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
    python tools/demo.py image -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/biker1.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
    ```

###  Tasks

- [ ] Setup Dataset and sample 100 images for each sample
- [ ] Training Loader updates
- [ ] Train the model 
- [ ] Test & Evaluate model
- [ ] Write report