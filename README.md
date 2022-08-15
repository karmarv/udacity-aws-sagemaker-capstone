# Capstone project (Object Detection)

### Objective 
- Detect cyclist in an image using object detection vision task.

### Tasks 

- [x] Submitted [proposal](./report/proposal.pdf)
- [x] Setup Dataset [coco-cyclist.zip](https://github.com/karmarv/udacity-aws-sagemaker-capstone/releases/download/alpha/coco-cyclist.zip)
    - Sample ~1000 images for bicycle and person in image
    - Use the same class_ids as in COCO_CLASSES. Provide additional class identifier label `cyclist` where both exist with IoU>0.25
    - Split Train/Val annotations and create COCO dataset
- [x] Training Loader updates in YOLOX (search for `_custom.py` in exps and yolox data/datasets folders)
- [x] Train the model (YOLOX nano, small and medium)
- [x] Test & Evaluate model [Weights & Bias Dashboard](https://wandb.ai/karmar/yolox?workspace=user-karmar)
- [x] Write [report](./report/report.pdf)

Next we have the process needed to replicate the experiments and showcase results. These cover the following:
- A. Environment
- B. Setup Training Dataset
- C. YOLOX Object Detector
- D. Detection Results
- E. W&B Training Visualizations

--- 

### A. Environment
- Install [conda environment](https://docs.conda.io/projects/conda/en/latest/glossary.html#silent-mode-glossary)
    -   ```bash
        conda env remove -n cap
        conda env create -n cap --file environment.yml --force
        conda activate cap 
        pip install fiftyone[desktop]
        ```

### B. Setup Training Dataset
- Dataset customization tutorial - [YOLOX Training Dataset](https://yolox.readthedocs.io/en/latest/train_custom_data.html)
- Download the [Cyclist coco-cyclist.zip dataset](https://github.com/karmarv/udacity-aws-sagemaker-capstone/releases/download/alpha/coco-cyclist.zip) in case you want to skip the dataset creation step and jump to detector evaluation.
    - unzip this data and then later move it to YOLOX dataset directory or create a soft link to this path.
- Generate sampled dataset with images for [`cyclist`,`person`,`bicycle`]
    - Download COOCO dataset to $COCO_DATA_HOME folder
    - Run the notebook [/samples/coco_sampler.ipynb](https://github.com/karmarv/udacity-aws-sagemaker-capstone/blob/main/samples/coco_sampler.ipynb)
    - Drop the sampled data folder to YOLOX/datasets folder
    - Configure the `/exps/default/yolox_nano_custom.py` with data_dir and json file paths
    - Setup the num_classes=3
- Visualize annotations with [FiftyOne tool](./samples/coco_fiftyone.py)
    - `python coco_fiftyone.py`
    - ![FiftyOne Data](./samples/data/cocobi/dataset-fiftyone.png?raw=true "FiftyOne Data")


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
        python tools/demo.py image -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/data/test/biker1.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```
    -   ```bash
        python tools/demo.py video -f ./exps/default/yolox_nano.py -c ./yolox_nano.pth --path ../samples/data/test/cyclists.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
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
- Train setup
    -   ```bash
        cd $YOLOX_HOME
        ln -s $COCO_HOME ./datasets/cocobi
        python -m yolox.tools.train -f ./exps/default/yolox_m_custom.py --devices 1 --batch-size 4 --fp16  --logger wandb wandb-project yolox
        ```
    -   ```bash 
        2022-08-13 21:34:32.943 | INFO     | yolox.core.trainer:before_train:130 - args: Namespace(batch_size=4, cache=False, ckpt=None, devices=1, dist_backend='nccl', dist_url=None, exp_file='./exps/default/yolox_m_custom.py', experiment_name='yolox_m_custom', fp16=True, logger='wandb', machine_rank=0, name=None, num_machines=1, occupy=False, opts=['wandb-project', 'yolox'], resume=False, start_epoch=None)
        2022-08-13 21:34:32.944 | INFO     | yolox.core.trainer:before_train:131 - exp value:
        ...
        2022-08-13 21:34:33.170 | INFO     | yolox.core.trainer:before_train:137 - Model Summary: Params: 8.93M, Gflops: 22.35
        2022-08-13 21:34:34.693 | INFO     | yolox.data.datasets.coco_custom:__init__:64 - loading annotations into memory...
        2022-08-13 21:34:34.786 | INFO     | yolox.data.datasets.coco_custom:__init__:64 - Done (t=0.09s)
        2022-08-13 21:34:34.786 | INFO     | pycocotools.coco:__init__:86 - creating index...
        2022-08-13 21:34:34.794 | INFO     | pycocotools.coco:__init__:86 - index created!
        2022-08-13 21:34:35.310 | INFO     | yolox.core.trainer:before_train:155 - init prefetcher, this might take one minute or less...
        2022-08-13 21:34:35.752 | INFO     | yolox.data.datasets.coco_custom:__init__:64 - loading annotations into memory...
        2022-08-13 21:34:35.757 | INFO     | yolox.data.datasets.coco_custom:__init__:64 - Done (t=0.00s)
        2022-08-13 21:34:35.757 | INFO     | pycocotools.coco:__init__:86 - creating index...
        2022-08-13 21:34:35.758 | INFO     | pycocotools.coco:__init__:86 - index created!
        2022-08-13 21:34:51.483 | INFO     | yolox.core.trainer:before_train:191 - Training start...
        ...
        2022-08-13 21:34:51.498 | INFO     | yolox.core.trainer:before_epoch:203 - ---> start train epoch1
        2022-08-13 21:34:54.633 | INFO     | yolox.core.trainer:after_iter:261 - epoch: 1/200, iter: 10/661, mem: 5707Mb, iter_time: 0.313s, data_time: 0.001s, total_loss: 18.7, iou_loss: 4.8, l1_loss: 0.0, conf_loss: 13.0, cls_loss: 0.9, lr: 5.722e-09, size: 640, ETA: 11:30:01
        2022-08-13 21:34:57.308 | INFO     | yolox.core.trainer:after_iter:261 - epoch: 1/200, iter: 20/661, mem: 5707Mb, iter_time: 0.267s, data_time: 0.001s, total_loss: 12.9, iou_loss: 4.8, l1_loss: 0.0, conf_loss: 7.3, cls_loss: 0.8, lr: 2.289e-08, size: 544, ETA: 10:39:03
        2022-08-13 21:35:00.414 | INFO     | yolox.core.trainer:after_iter:261 - epoch: 1/200, iter: 30/661, mem: 5707Mb, iter_time: 0.310s, data_time: 0.001s, total_loss: 17.7, iou_loss: 4.8, l1_loss: 0.0, conf_loss: 12.1, cls_loss: 0.8, lr: 5.150e-08, size: 672, ETA: 10:53:4
        ... 
        
        2022-08-14 03:20:40.437 | INFO     | yolox.core.trainer:after_iter:261 - epoch: 200/200, iter: 650/661, mem: 5707Mb, iter_time: 0.162s, data_time: 0.001s, total_loss: 5.7, iou_loss: 2.3, l1_loss: 1.1, conf_loss: 1.8, cls_loss: 0.6, lr: 3.125e-05, size: 672, ETA: 0:00:01
        2022-08-14 03:20:42.162 | INFO     | yolox.core.trainer:after_iter:261 - epoch: 200/200, iter: 660/661, mem: 5707Mb, iter_time: 0.172s, data_time: 0.001s, total_loss: 5.4, iou_loss: 2.1, l1_loss: 1.3, conf_loss: 1.4, cls_loss: 0.6, lr: 3.125e-05, size: 736, ETA: 0:00:00
        2022-08-14 03:20:42.352 | INFO     | yolox.core.trainer:save_ckpt:356 - Save weights to ./YOLOX_outputs/yolox_m_custom
        2022-08-14 03:20:44.486 | INFO     | yolox.evaluators.coco_evaluator:evaluate_prediction:256 - Evaluate in main process...
        2022-08-14 03:20:46.786 | INFO     | yolox.core.trainer:evaluate_and_save_model:346 - 
        Average forward time: 8.33 ms, Average NMS time: 0.51 ms, Average inference time: 8.83 ms
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.258
         Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
         Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.217
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
         Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.353
         Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.370
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.191
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.304
         Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.495
         Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.571
        
        2022-08-14 03:20:46.786 | INFO     | yolox.core.trainer:save_ckpt:356 - Save weights to ./YOLOX_outputs/yolox_m_custom
        2022-08-14 03:20:47.059 | INFO     | yolox.core.trainer:save_ckpt:356 - Save weights to ./YOLOX_outputs/yolox_m_custom
        2022-08-14 03:20:47.200 | INFO     | yolox.core.trainer:after_train:196 - Training of experiment is done and the best AP is 25.82
        ```
- Tests
    -   ```bash
        python tools/demo_custom.py video -f ./exps/default/yolox_nano_custom.py -c ./YOLOX_outputs/yolox_nano_custom/best_ckpt.pth --path ../samples/data/test/bike-road.mp4 --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu
        ```

- Experiments
    - Yolox nano, batch 16, epoch 100   
        ```bash
        python -m yolox.tools.train -f ./exps/default/yolox_nano_custom.py --devices 1 --batch-size 16 --fp16  --logger wandb wandb-project yolox
        ```
    - Yolox medium, batch 4, epoch 200
        ```bash 
        python -m yolox.tools.train -f ./exps/default/yolox_m_custom.py --devices 1 --batch-size 4 --fp16  --logger wandb wandb-project yolox
        ```

## D. Detection Results

> Biker image output from a YOLOX-nano model 

- ![Biker Image](./samples/data/test/output/biker1.jpg?raw=true "Detected Biker")

> Attempts a video output from a YOLOX-nano model (Click on the link to preview)

- [![Output Biker Kid Video](./samples/data/test/output/bike-kid.gif)](./samples/data/test/output/bike-kid.mp4?raw=true "Detected Biker Kid Video")
 [![Output Road Biker Video](./samples/data/test/output/bike-road.gif)](./samples/data/test/output/bike-road.mp4?raw=true "Detected Road Biker Video")

## E. W&B Training Visualizations

> Track progress on W&B dashboard during training 

- ![Dashboard Image](./samples/data/test/output/wandb-dashboard.png?raw=true "Dashboard")

> Final AP metrics in W&B dashboard

- ![Dashboard Final](./samples/data/test/output/wandb-dashboard-final.png?raw=true "Dashboard Final")

