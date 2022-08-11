import os
import copy

import cv2
import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset



def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)

class COCODatasetCustom(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        selected_categories=[]
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        #super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        in_json_file  = os.path.join(self.data_dir, "annotations", self.json_file)
        self.coco = COCO(in_json_file)
        #remove_useless_info(self.coco)
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.selected_categories = selected_categories
        
        # Fetch class IDs only corresponding to the filterClasses
        if len(selected_categories)>0:
            self.class_ids = sorted(self.coco.getCatIds(catNms=self.selected_categories))
            self.cats = self.coco.loadCats(self.class_ids)
            # Get all images containing the above Category IDs
            self.ids = self.coco.getImgIds(catIds=self.class_ids)
        else:
            self.class_ids = sorted(self.coco.getCatIds())
            self.cats = self.coco.loadCats(self.coco.getCatIds())
            self.ids = self.coco.getImgIds()
        
        self._classes = tuple([c["name"] for c in self.cats])
        self.annotations = self._load_coco_annotations()
        print("Read annotations for {} images from -> {}".format(len(self.ids), in_json_file))


    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]
    
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["category_id"] in self.class_ids and obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            if obj["category_id"] in self.class_ids:
                cls = self.class_ids.index(obj["category_id"])
                res[ix, 0:4] = obj["clean_bbox"]
                res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img
    
    def export_anns(self, out_filepath, do_update=False):
        import json
        # For the image ids export the annotations of interest
        image_ids = self.ids
        anno_ids  = self.coco.getAnnIds(imgIds=image_ids, iscrowd=False)
        anno_id_max = max(anno_ids)+1
        categories = self.cats
        annotations = []
        # Remove other annotations in the same image
        for obj in self.coco.loadAnns(anno_ids):
            if obj["category_id"] in self.class_ids:
                obj.pop("segmentation", None)
                annotations.append(obj)

        if do_update:
            categories, updated_annotations = self.update_anns(anno_id_max)
            print("New Annotations: {}/{}".format(len(updated_annotations), len(annotations)))
            annotations.extend(updated_annotations)

        anns_dict = { "categories": categories, "images": self.coco.loadImgs(image_ids), "annotations": annotations }
        print("Export {} annotations for {} images to -> {}".format(len(annotations), len(image_ids), out_filepath))
        with open(out_filepath, "w") as outfile:
            json.dump(anns_dict, outfile, indent=4,  sort_keys=True)
    
    
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # determine union bbox        
        xuA = min(boxA[0], boxB[0])
        yuA = min(boxA[1], boxB[1])
        xuB = max(boxA[2], boxB[2])
        yuB = max(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou, [xuA,yuA,xuB,yuB]
    
    
    def update_anns(self, anno_id_max):
        # For the image annotations where person and bicycle have overlap
        # Add a new category
        cyclist_class_id = 101
        updated_categories = self.cats
        updated_categories.append({'supercategory': 'person', 'id': cyclist_class_id, 'name': 'cyclist'})
        
        def get_bbox(obj, width, height):
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            return [x1,y1,x2,y2]
        
        # Add a new annotation named cyclist = person+bicycle
        image_ids = self.ids
        updated_annotations = []
        for id_ in image_ids:
            im_ann = self.coco.loadImgs(id_)[0]
            width = im_ann["width"]
            height = im_ann["height"]
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
            annotations = self.coco.loadAnns(anno_ids)
            # Sample Person and compare if there are any IoU with bicycle
            anno_idx_persons = [ idx for idx, obj in enumerate(annotations) if obj["category_id"]==1]
            for idx_person in anno_idx_persons:
                for idx, obj in enumerate(annotations):
                    obj["clean_bbox"] = get_bbox(obj, width, height)
                    if obj["category_id"] in self.class_ids and obj["area"] > 0:
                        if idx != idx_person and obj["category_id"]==2:
                            # Try to find IoU with Bicycles
                            ann_person = annotations[idx_person]
                            box_person = get_bbox(ann_person, width, height)
                            box_cycle = obj["clean_bbox"]
                            iou, bbox_union = self.bb_intersection_over_union(box_person, box_cycle)
                            if iou > 0.25:
                                #print("({}.)\t IoU:{} \t Person:{}, \t Cycle:{}".format(obj["id"],iou, ann_person["id"], obj["id"]))                           
                                # Append a new entry (high IoU) for this bicyclist=120
                                new_ann = copy.deepcopy(ann_person)
                                new_ann["clean_bbox"]  = bbox_union
                                new_ann["bbox"]        = bbox_union
                                new_ann["category_id"] = cyclist_class_id
                                new_ann["id"]          = anno_id_max + idx
                                updated_annotations.append(new_ann)
                                #print("\tNew Annotation: ", new_ann)
        return updated_categories, updated_annotations
    
    
    def show_anns(self):
        from matplotlib import pyplot as plt
        index = np.random.randint(0,len(self.ids))
        img = self.coco.loadImgs(self.ids[index])[0]
        # Load and display instance annotations
        plt.imshow(self.load_image(index))
        plt.axis('off')
        annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.class_ids, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        self.coco.showAnns(anns)
        for item in anns:
            print("Annotations: ", item['category_id'], item['id'], item['bbox'])
        plt.show()

    
if __name__ == "__main__":
    COCO_DATA_HOME = "/media/rahul/HIT-GRAID/data/coco"
    ann_file       = "instances_val2017.json"
    ann_type       = "val2017"
    allowed_list   = ["person", "bicycle"]
    coco_custom    = COCODatasetCustom(data_dir=COCO_DATA_HOME, json_file=ann_file, name=ann_type, selected_categories=allowed_list)   
    print("Number of images containing all the  classes:", len(coco_custom.ids))
    #coco_custom.show_anns()
    coco_custom.export_anns(os.path.join("./annotations", "selected_instances_val2017.json"), True)
    