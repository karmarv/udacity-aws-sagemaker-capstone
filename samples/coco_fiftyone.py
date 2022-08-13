import os
from wsgiref.simple_server import WSGIRequestHandler

import fiftyone as fo
import fiftyone.zoo as foz


"""
Link: https://github.com/voxel51/fiftyone

pip install fiftyone
pip install fiftyone[desktop]

"""

def load_custom_coco(ann_type="train2017"):
    TARGET_COCO_DATA_HOME = "./data/cocobi"
    TRAIN_IMAGE_DIR = os.path.join(TARGET_COCO_DATA_HOME, ann_type)
    TRAIN_ANNS = os.path.join(TARGET_COCO_DATA_HOME, "annotations", "sampled_instances_{}.json".format(ann_type))
    # Load COCO formatted dataset
    coco_train_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=TRAIN_IMAGE_DIR,
        labels_path=TRAIN_ANNS
    )
    # Verify that the class list for our dataset was imported
    print(coco_train_dataset.default_classes)  # ['airplane', 'apple', ...]
    print(coco_train_dataset)
    session = fo.launch_app(coco_train_dataset, desktop=True)
    session.wait()

def load_zoo_coco():
    dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                label_types=["detections"],
                classes=["person","bicycle"],
                max_samples=50,
                )
    session = fo.launch_app(dataset, desktop=True)
    session.wait()

load_custom_coco(ann_type="val2017")