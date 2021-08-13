from typing import Any, Dict, List
import json
import h5py
import numpy as np
from tqdm import tqdm
import random
from anytree import AnyNode
from anytree.search import findall_by_attr,findall
import copy
import string

class OIDictImporter(object):
    ''' Importer that works on Open Images json hierarchy '''
    def __init__(self, nodecls=AnyNode):
        self.nodecls = nodecls

    def import_(self, data):
        """Import tree from `data`."""
        return self.__import(data)


    def __import(self, data, parent=None):
        assert isinstance(data, dict)
        assert "parent" not in data
        attrs = dict(data)
        children = attrs.pop("Subcategory", [])
        node = self.nodecls(parent=parent, **attrs)
        for child in children:
            self.__import(child, parent=node)
        return node

class HierarchyFinder(object):

    def __init__(self, class_structure_path, abstract_list_path):
        importer = OIDictImporter()
        with open(class_structure_path) as f:
            self.class_structure = importer.import_(json.load(f))

        with open(abstract_list_path) as out:
            self.abstract_list = json.load(out)

    def find_key(self, label):
        if label in self.abstract_list:
            return label
        return None

    def find_parent(self, label):
        target_node = findall(self.class_structure, filter_=lambda node: node.LabelName.lower() in (label))[0]
        while  self.find_key(target_node.LabelName.lower()) is None:
            target_node = target_node.parent
        return self.find_key(target_node.LabelName.lower())

def nms(dets, classes, hierarchy, thresh=0.8):
    # Non-max suppression of overlapping boxes where score is based on 'height' in the hierarchy,
    # defined as the number of edges on the longest path to a leaf
    scores = [findall(hierarchy, filter_=lambda node: node.LabelName.lower() == cls)[0].height for cls in classes]
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = np.array(scores)
    order = scores.argsort()

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # check the score, objects with smaller or equal number of layers cannot be removed.
        keep_condition = np.logical_or(scores[order[1:]] <= scores[i], \
            inter / (areas[i] + areas[order[1:]] - inter) <= thresh)

        inds = np.where(keep_condition)[0]
        order = order[inds + 1]

    return keep

class ImageFeaturesReader(object):
    r"""
    A reader for H5 files containing pre-extracted image features. A typical image features file
    should have at least two H5 datasets, named ``image_id`` and ``features``. It may optionally
    have other H5 datasets, such as ``boxes`` (for bounding box coordinates), ``width`` and
    ``height`` for image size, and others. This reader only reads image features, because our
    UpDown captioner baseline does not require anything other than image features.

    Example of an h5 file::

        image_bottomup_features.h5
        |--- "image_id" [shape: (num_images, )]
        |--- "features" [shape: (num_images, num_boxes, feature_size)]
        +--- .attrs {"split": "coco_train2017"}

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing image ids and features corresponding to one of the four
        ``split``s used: "coco_train2017", "coco_val2017", "nocaps_val", "nocaps_test".
    in_memory : bool
        Whether to load the features in memory. Beware, these files are sometimes tens of GBs
        in size. Set this to true if you have sufficient RAM.
    """

    def __init__(self, features_h5path: str, start_index: int = 0):
        self.features_h5path = features_h5path

        # Keys are all the image ids, values depend on ``self._in_memory``.
        # If ``self._in_memory`` is True, values are image features corresponding to the image id.
        # Else values will be integers; indices in the files to read features from.
        self._map: Dict[int, Union[int, np.ndarray]] = {}

        features_h5 = h5py.File(self.features_h5path, "r")
        self._width = features_h5["width"][:]
        self._height = features_h5["height"][:]
        self.start_index = start_index

        image_id_np = np.array(features_h5["image_id"])
        self._map = {
            image_id_np[index]: index for index in range(image_id_np.shape[0])
        }
            
        features_h5.close()

        self.features_h5 = None

    def open_h5_file(self):
        self.features_h5 = h5py.File(self.features_h5path, "r")

    def close_h5_file(self):
        self.features_h5.close()
        self.features_h5 = None

    def __len__(self):
        return len(self._map)

    def process_box(self, index, box_np, score):
        new_box_np = np.zeros((box_np.shape[0], 8), dtype=np.float32)

        if score.shape[0] > box_np.shape[0]:
            score = score[:box_np.shape[0]]

        box_np[:, 0] /= self._width[index]
        box_np[:, 2] /= self._width[index]
        box_np[:, 1] /= self._height[index]
        box_np[:, 3] /= self._height[index]
        if box_np.shape[0] > 0:
            new_box_np[:, :4] = box_np
            new_box_np[:, 4] =  box_np[:, 2] - box_np[:, 0]
            new_box_np[:, 5] = box_np[:, 3] - box_np[:, 1]
            new_box_np[:, 6] = (box_np[:, 2] - box_np[:, 0]) * (box_np[:, 3] - box_np[:, 1])
            min_size = min(score.shape[0], box_np.shape[0])
            new_box_np[:min_size, 7] = score[:min_size]

        return new_box_np

    def __getitem__(self, image_id: int):
        if self.features_h5 is None:
            features_h5 = h5py.File(self.features_h5path, "r")
        else:
            features_h5 = self.features_h5
        index = self._map[image_id]
        image_id_features = features_h5["features"][index].astype('float32').reshape(-1, 2048)
        class_ = features_h5["classes"][index].astype('int64')
        score = features_h5["scores"][index].reshape(-1)
        box_np = features_h5["boxes"][index].reshape(-1, 4)
        new_box_np = self.process_box(index, box_np, score)
        if self.features_h5 is None:
            features_h5.close()
        min_size = min([image_id_features.shape[0], new_box_np.shape[0], class_.shape[0]])
        return image_id_features[:min_size], new_box_np[:min_size], class_[:min_size] + self.start_index

class CocoCaptionsReader(object):
    def __init__(self, captions_jsonpath, captions_word_norm_jsonpath=None, rm_dumplicated_caption=False, shuffle=False, is_train=True, rm_punctuation=False):

        self._captions_jsonpath = captions_jsonpath

        with open(captions_jsonpath) as cap:
            captions_json = json.load(cap)

        vocab_norm = None
        if captions_word_norm_jsonpath is not None:
            with open(captions_word_norm_jsonpath) as word_norm:
                vocab_norm = json.load(word_norm)
        # List of (image id, caption) tuples.
        _captions_dict = {}
        caption_set = set()
        rm_dump_cap = 0
        c = copy.deepcopy
        print(f"Tokenizing captions from {captions_jsonpath}...")
        for caption_item in tqdm(captions_json["annotations"]):
            if 'unable' in caption_item["caption"]:
                continue
            if caption_item["caption"] in caption_set and rm_dumplicated_caption and is_train:
                rm_dump_cap += 1
                continue
            else:
                caption_set.add(caption_item["caption"])
            caption_item["gt"] = c(caption_item["caption"])
            caption_item["caption"]: str = caption_item["caption"].lower().strip()
            if rm_punctuation:
                for p in string.punctuation:
                    caption_item["caption"] = caption_item["caption"].replace(p, ' ')
            if vocab_norm is not None:
                for key, value in vocab_norm.items():
                    caption_item['caption'] = caption_item['caption'].replace(' ' + key + ' ', ' ' + value + ' ')
            if caption_item["image_id"] not in _captions_dict:
                _captions_dict[caption_item["image_id"]] = []
            _captions_dict[caption_item["image_id"]].append(caption_item)
        
        self._captions_together = []
        for (k, captions) in _captions_dict.items():
            gt = [c['gt'] for c in captions]
            for cap in  captions:
                self._captions_together.append((k, cap['caption'], gt))
        self._captions_dict = _captions_dict

        if rm_dumplicated_caption and is_train:
            print("remove duplicate captions %d" % rm_dump_cap)

        if shuffle:
            random.shuffle(self._captions_together)


    def __len__(self):
        return len(self._captions_together)

    def __getitem__(self, index):
        img_id, cap, gt = self._captions_together[index]
        return img_id, cap, gt

    def get_gt_by_image_id(self, image_id):
        caps = self._captions_dict[image_id]
        return [c['gt'] for c in caps]

class BoxesReader(object):
    """
    A reader for H5 files containing bounding boxes, classes and confidence scores inferred using
    an object detector. A typical H5 file should at least have the following structure:
    ```
    image_boxes.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "width" [shape: (num_images, )]
       |--- "height" [shape: (num_images, )]
       |--- "boxes" [shape: (num_images, max_num_boxes, 4)]
       |--- "classes" [shape: (num_images, max_num_boxes, )]
       +--- "scores" [shape: (num_images, max_num_boxes, )]
    ```
    Box coordinates are of form [X1, Y1, X2, Y2], _not_ normalized by image width and height. Class
    IDs start from 1, i-th ID corresponds to (i-1)-th category in "categories" field of
    corresponding annotation file for this split (in COCO format).
    Parameters
    ----------
    boxes_h5path : str
        Path to an H5 file containing boxes, classes and scores of a particular dataset split.
    """

    def __init__(self, 
                boxes_h5path: str, 
                detection_dict: Dict[int, str], 
                object_blacklist_path: str, 
                class_structure_path: str = None, 
                abstract_list_path: str = None,
                min_score: float = 0.01, 
                top_k: int = 3,
                is_val: bool = False, 
                cls_start_index: int = 0,
                object_filtering: bool = True,
                variant_copy_candidates: bool = True,
                in_memory: bool = False,
                copy_candidate_clear_up=False):
    
        with open(object_blacklist_path) as out:
            blacklist = json.load(out)
            full_list = blacklist['blacklist_categories'] + (blacklist['val_blacklist_categories'] if is_val else [])
        self._blacklist_categories = set([s.lower() for s in full_list])
        self._boxes_h5path = boxes_h5path
        self.detection_dict = detection_dict
        self.min_score = min_score
        self.is_val = is_val
        self.object_filtering = object_filtering
        self.top_k = top_k
        self.cls_start_index = cls_start_index
        self.in_memory = in_memory
        self.variant_copy_candidates = variant_copy_candidates
        self.copy_candidate_clear_up = copy_candidate_clear_up

        if abstract_list_path is not None and class_structure_path is not None:
            self.hierarchy_finder = HierarchyFinder(class_structure_path, abstract_list_path)
        else:
            self.hierarchy_finder = None

        self.cache = {}
 
        with h5py.File(self._boxes_h5path, "r") as boxes_h5:
            self._width = boxes_h5["width"][:]
            self._height = boxes_h5["height"][:]
            self._image_ids = boxes_h5["image_id"][:].tolist()
            self._image_ids = {
                image_id: index for index, image_id in enumerate(self._image_ids)
            }

            if self.in_memory:
                for image_id in tqdm(self._image_ids):
                    self.process_single_image(image_id, self._image_ids[image_id], boxes_h5)
        
    def __len__(self):
        return len(self._image_ids)

    def process_single_image(self, image_id, i, boxes_h5):
        feature = boxes_h5["features"][i].reshape(-1, 2048)
        box_np = boxes_h5["boxes"][i].reshape(-1, 4)
        box_score = boxes_h5["scores"][i]
        class_list = (boxes_h5["classes"][i] + self.cls_start_index).tolist()

        new_box_np = np.zeros((box_np.shape[0], 8))
        if box_np.shape[0] > 0:
            box_np[:, 0] /= self._width[i]
            box_np[:, 2] /= self._width[i]
            box_np[:, 1] /= self._height[i]
            box_np[:, 3] /= self._height[i]

            new_box_np[:, :4] = box_np
            new_box_np[:, 4] =  box_np[:, 2] - box_np[:, 0]
            new_box_np[:, 5] = box_np[:, 3] - box_np[:, 1]
            new_box_np[:, 6] = (box_np[:, 2] - box_np[:, 0]) * (box_np[:, 3] - box_np[:, 1])

            min_size = min(box_score.shape[0], box_np.shape[0])
            new_box_np[:min_size, 7] = box_score[:min_size]

        feature = feature[:box_np.shape[0]]
        new_box_np = new_box_np[:box_np.shape[0]]
        box_score = box_score[:box_np.shape[0]]
        class_list = class_list[:box_np.shape[0]]

        if not self.variant_copy_candidates:
            _class = []
            _box = []
            _feature = []

            min_size = min(feature.shape[0], new_box_np.shape[0])
            feature = feature[:min_size]
            new_box_np = new_box_np[:min_size]
            box_score = box_score[:min_size]
            class_list = class_list[:min_size]

            for idx, box_cls in enumerate(class_list):
                if box_cls not in self.detection_dict or box_score[idx] < self.min_score: 
                    continue
                _box.append(new_box_np[idx])
                _class.append(box_cls)
                _feature.append(feature[idx])
            
            new_box_np = np.zeros((len(_box), 8))
            for i, bb in enumerate(_box):
                new_box_np[i] = bb
            feature_np = np.array(_feature)
            class_np = np.array(_class)
            obj_mask = np.ones((len(_class),), dtype=np.float32)
            
            for idx, box_cls in enumerate(class_np):
                text_class = self.detection_dict[box_cls]
                if text_class in self._blacklist_categories:
                    obj_mask[idx] = 0.0

            if self.object_filtering:
                keep = nms(new_box_np, [self.detection_dict[box_cls] for box_cls in _class], self.hierarchy_finder.class_structure)
                for idx in range(len(_class)):
                    if idx not in keep:
                        obj_mask[idx] = 0.0

            if new_box_np.shape[0] > 0:
                anns = []
                for idx, (box, cls_, mask) in enumerate(zip(new_box_np, class_np, obj_mask)):
                    if mask == 1.0:
                        anns.append((box, cls_, idx))

                anns = sorted(anns, key=lambda x:x[0][7], reverse=True)

                if self.object_filtering:
                    anns = anns[:self.top_k]

                seen_class = []
                for box, cls_, idx in anns:
                    if cls_ not in seen_class:
                        seen_class.append(cls_)
                        obj_mask[idx] = 2.0

                obj_mask[obj_mask < 2.0] = 0.0
                obj_mask[obj_mask == 2.0] = 1.0

            class_list = class_np.tolist()
            text_class = [self.detection_dict[v] for v in class_list]
            if self.hierarchy_finder is not None:
                parent_class = [self.hierarchy_finder.find_parent(v) for v in text_class]
                parent_class_index = [self.hierarchy_finder.abstract_list[v] for v in parent_class]
            else:
                parent_class_index = [0 for _ in range(len(text_class))]

            new_box_np = new_box_np.astype('float32')
            class_np = np.array(class_list).astype('int64')
            feature_np = feature_np.astype('float32')
            parent_class_np = np.array(parent_class_index).astype('int64')

            if not self.copy_candidate_clear_up:
                self.cache[image_id] = {
                    "predicted_boxes": new_box_np,
                    "predicted_classes": class_np,
                    "predicted_feature": feature_np,
                    "parent_classes": parent_class_np,
                    "predicted_mask":  obj_mask
                }
            else:
                self.cache[image_id] = {
                    "predicted_boxes": new_box_np[obj_mask == 1],
                    "predicted_classes": class_np[obj_mask == 1],
                    "predicted_feature": feature_np[obj_mask == 1],
                    "parent_classes": parent_class_np[obj_mask == 1],
                    "predicted_mask":  obj_mask[obj_mask == 1]
                }
        else:
            text_class = [self.detection_dict[v] for v in class_list]
            if self.hierarchy_finder is not None:
                parent_class = [self.hierarchy_finder.find_parent(v) for v in text_class]
                parent_class_index = [self.hierarchy_finder.abstract_list[v] for v in parent_class]
            else:
                parent_class_index = [0 for _ in range(len(text_class))]

            new_box_np = new_box_np.astype('float32')
            class_np = np.array(class_list).astype('int64')
            feature_np = feature.astype('float32')
            parent_class_np = np.array(parent_class_index).astype('int64')
            obj_mask = np.ones_like(class_np)

            if not self.copy_candidate_clear_up:
                self.cache[image_id] = {
                    "predicted_boxes": new_box_np,
                    "predicted_classes": class_np,
                    "predicted_feature": feature_np,
                    "parent_classes": parent_class_np,
                    "predicted_mask":  obj_mask
                }
            else:
                self.cache[image_id] = {
                    "predicted_boxes": new_box_np[obj_mask == 1],
                    "predicted_classes": class_np[obj_mask == 1],
                    "predicted_feature": feature_np[obj_mask == 1],
                    "parent_classes": parent_class_np[obj_mask == 1],
                    "predicted_mask":  obj_mask[obj_mask == 1]
                }



    def __getitem__(self, image_id: int):
        i = self._image_ids[image_id]
        
        if image_id in self.cache:
            return self.cache[image_id]

        with h5py.File(self._boxes_h5path, "r") as boxes_h5:
            self.process_single_image(image_id, i, boxes_h5)

        d = self.cache[image_id]
        return {key: np.array(d[key], copy=True) for key in d}
