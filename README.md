# Udacity AWS Sagemaker Capstone
AWS Sagemaker Tutorial Capstone project

## Objective: Detect cyclist in an image using object detection vision task.


###  Tasks

- [x] Submitted [proposal](./report/proposal.pdf)
- [x] Setup Dataset [coco-cyclist.zip](https://github.com/karmarv/udacity-aws-sagemaker-capstone/releases/download/alpha/coco-cyclist.zip)
    - Sample ~1000 images for bicycle and person in image
    - Use the same class_ids as in COCO_CLASSES. Provide additional class identifier label `cyclist` where both exist with IoU>0.25
    - Split Train/Val annotations and create COCO dataset
- [x] Training Loader updates in YOLOX (search for `_custom.py` in exps and yolox data/datasets folders)
- [x] Train the model (YOLOX nano, small and medium)
- [x] Test & Evaluate model [Weights & Bias Dashboard](https://wandb.ai/karmar/yolox?workspace=user-karmar)
- [x] Write [report](./report/report.pdf)


### A. Environment
- Install [conda environment](https://docs.conda.io/projects/conda/en/latest/glossary.html#silent-mode-glossary)
    -   ```bash
        conda env remove -n cap
        conda env create -n cap --file environment.yml --force
        conda activate cap 
        ```

### B. Setup [Training Dataset](https://yolox.readthedocs.io/en/latest/train_custom_data.html)
- Download the [Cyclist coco-cyclist.zip dataset](https://github.com/karmarv/udacity-aws-sagemaker-capstone/releases/download/alpha/coco-cyclist.zip) in case you want to skip the dataset creation step and jump to detector evaluation.
    - unziop this data and then later move it to YOLOX dataset directory or create a soft link to this path.
- Generate sampled dataset with images for [`cyclist`,`person`,`bicycle`]
    - Download COOCO dataset to $COCO_DATA_HOME folder
    - Run the notebook [/samples/coco_sampler.ipynb](https://github.com/karmarv/udacity-aws-sagemaker-capstone/blob/main/samples/coco_sampler.ipynb)
    - Drop the sampled data folder to YOLOX/datasets folder
    - Configure the `/exps/default/yolox_nano_custom.py` with data_dir and json file paths
    - Setup the num_classes=3
- Visualize annotations with [FiftyOne tool](./samples/coco_fiftyone.py)
    - `python coco_fiftyone.py`



### C. YOLOX Object Detector

- Install [YoloX](https://github.com/Megvii-BaseDetection/YOLOX/)
    -   ```bash
        git clone --recurse-submodules https://github.com/karmarv/udacity-aws-sagemaker-capstone
        export YOLOX_HOME=~/udacity-aws-sagemaker-capstone/YOLOX
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
- Setup Dataset loader with `cyclist`,`person`,`bicycle` class
    - Customize the [yolox/data/datasets/coco.py](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/coco.py) to load the classess and images of interest [`/data/datasets/coco_cyclist.py`](https://github.com/karmarv/YOLOX/blob/main/yolox/data/datasets/coco_custom.py)
- Create experiment file to control data loading and model training [`/exps/default/yolox_nano_custom.py`](https://github.com/karmarv/YOLOX/blob/main/exps/default/yolox_nano_custom.py)
    - See get_data_loader, get_eval_loader, and get_evaluator for more details
- Demo Test for Image - [Documentation](https://yolox.readthedocs.io/en/latest/quick_run.html)
    -   ```bash
        python tools/demo.py image -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/biker1.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```
    -   ```bash
        python tools/demo.py video -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/cyclists.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```
- Dataset [Created `cyclist` for Detection 2017](https://cocodataset.org/#download)
    - Download `COCO Cyclist` Train/Val annotation and images dataset
        - Link: https://github.com/karmarv/udacity-aws-sagemaker-capstone/releases/download/alpha/coco-cyclist.zip 
    -   ```bash
        cocobi/
            annotations/*.json           # contains annotation label as instances_{train,val}2017.json
            {train,val}2017/*.jpg        # contains image files that are mentioned in the corresponding json
        ```
    -   ```bash
        export COCO_HOME=~/data/cocobi
        ```
- Train setup test
    -   ```bash
        cd $YOLOX_HOME
        ln -s $COCO_HOME ./datasets/cocobi
        python -m yolox.tools.train -f ./exps/default/yolox_nano_custom.py -d 1 -b 8 --fp16  --logger wandb wandb-project yolox
        ```
    -   ```bash 
        2022-08-07 20:22:27 | INFO     | yolox.core.trainer:137 - Model Summary: Params: 0.91M, Gflops: 1.11
        2022-08-07 20:22:28 | INFO     | yolox.data.datasets.coco:64 - loading annotations into memory...
        2022-08-07 20:22:37 | INFO     | yolox.data.datasets.coco:64 - Done (t=8.70s)
        2022-08-07 20:22:37 | INFO     | pycocotools.coco:86 - creating index...
        2022-08-07 20:22:37 | INFO     | pycocotools.coco:86 - index created!
        2022-08-07 20:23:01 | INFO     | yolox.core.trainer:155 - init prefetcher, this might take one minute or less...
        2022-08-07 20:23:04 | INFO     | yolox.data.datasets.coco:64 - loading annotations into memory...
        2022-08-07 20:23:04 | INFO     | yolox.data.datasets.coco:64 - Done (t=0.36s)
        2022-08-07 20:23:04 | INFO     | pycocotools.coco:86 - creating index...
        2022-08-07 20:23:04 | INFO     | pycocotools.coco:86 - index created!
        wandb: Currently logged in as: karmar. Use `wandb login --relogin` to force relogin
        wandb: Tracking run with wandb version 0.13.1
        wandb: Run data is saved locally in /home/rahul/workspace/coursera-sdsc-words/udacity-aws-sagemaker-capstone/YOLOX/wandb/run-20220807_202306-1p4lrdls
        wandb: Run `wandb offline` to turn off syncing.
        wandb: Syncing run cosmic-field-2
        wandb:  View project at https://wandb.ai/karmar/yolox
        wandb:  View run at https://wandb.ai/karmar/yolox/runs/1p4lrdls
        2022-08-07 20:23:26 | INFO     | yolox.core.trainer:191 - Training start...
        2022-08-07 20:23:26 | INFO     | yolox.core.trainer:192 - 
        ....
        2022-08-07 20:23:26 | INFO     | yolox.core.trainer:203 - ---> start train epoch1
        2022-08-07 20:23:28 | INFO     | yolox.core.trainer:261 - epoch: 1/300, iter: 10/14786, mem: 584Mb, iter_time: 0.224s, data_time: 0.002s, total_loss: 14.6, iou_loss: 4.6, l1_loss: 0.0, conf_loss: 7.8, cls_loss: 2.1, lr: 2.287e-11, size: 416, ETA: 11 days, 11:38:15
        ...
        2022-08-07 21:16:42 | INFO     | yolox.core.trainer:356 - Save weights to ./YOLOX_outputs/yolox_nano
        2022-08-07 21:16:42 | INFO     | yolox.core.trainer:203 - ---> start train epoch2
        2022-08-07 21:16:44 | INFO     | yolox.core.trainer:261 - epoch: 2/300, iter: 10/14786, mem: 1708Mb, iter_time: 0.222s, data_time: 0.108s, total_loss: 11.5, iou_loss: 3.8, l1_loss: 0.0, conf_loss: 5.2, cls_loss: 2.4, lr: 5.007e-05, size: 512, ETA: 11 days, 0:55:22
        ...
        ```
- Tests
    -   ```bash
        python tools/demo_custom.py video -f ./exps/default/yolox_nano_custom.py -c ./YOLOX_outputs/yolox_nano_custom/best_ckpt.pth --path ../samples/data/test/bike-road.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```

- Expected Experiments
    - Yolox nano, batch 16, epoch 100   
        ```bash
        python -m yolox.tools.train -f ./exps/default/yolox_nano_custom.py --devices 1 --batch-size 16 --fp16  --logger wandb wandb-project yolox
        ```
    - Yolox medium, batch 4, epoch 200
        ```bash 
        python -m yolox.tools.train -f ./exps/default/yolox_m_custom.py --devices 1 --batch-size 4 --fp16  --logger wandb wandb-project yolox
        ```

## Sample Result

> Biker image output from a YOLOX-nano model 

![Biker Image](./samples/data/test/output/biker1.jpg?raw=true "Detected Biker")

> Attempts a video output from a YOLOX-nano model

![Biker Video](./samples/data/test/output/bike-kid.mp4?raw=true "Detected Biker Video")

## Visualize Training 

> Track progress on W&B dashboard

![Dashboard Image](./samples/data/test/output/wandb-dashboard.png?raw=true "Dashboard")

