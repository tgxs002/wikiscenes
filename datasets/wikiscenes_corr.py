import os, torch, json, random, time
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from .utils import colormap
import datasets.transforms as tf
import _pickle as cPickle
import torchvision.transforms.functional as F
from torchvision.utils import save_image as sv
from collections import Counter

classes = [
        'facade', 'window', 'chapel', 'organ', 'nave', 'tower', 'choir', 'portal', 'altar', 'statue'
    ]

def load_dataset(root, _split_f, corr):
    images = []
    labels = []
    captions = []
    tags_list = []
    keypoints = []
    # we do not have ground truth
    masks = None
    corr_index = dict()
    with open(_split_f, "r") as lines:
        print("building graph")
        count = 0
        for line in lines:
            _image, label, caption, tags = line.strip("\n").split(':')
            image_corr = _image[22:]
            if image_corr in corr:
                k = corr[image_corr]
                for c in k:
                    p = k[c]
                    if c in corr_index:
                        corr_index[c].add(count)
                    else:
                        corr_index[c] = set([count])
                    k[c] = (p[1], p[0])
                keypoints.append(k)
            else:
                keypoints.append(None)
            _image = os.path.join(root, _image)
            assert os.path.isfile(_image), '%s not found' % _image
            # if os.path.isfile(_image):
            images.append(_image)
            labels.append([x.strip("',") for x in label.strip("[]").split()])
            captions.append(caption)
            tags_list.append(tags)
            count += 1

    print("Filtering")
    image_graph = dict()
    for i, keypoint in enumerate(keypoints):
        if keypoint != None:
            t = Counter()
            for p in keypoint:
                t.update(corr_index[p])
            t = [key for key, cnt in t.items() if cnt >= 10]
            if t:
                image_graph[i] = t

    return images, labels, captions, tags_list, keypoints, masks, corr_index, image_graph


class WikiScenes_corr(Dataset):

    dataset_classes = classes

    CLASSES = ["background"]
    CLASSES += dataset_classes
    CLASSES.append("ambiguous")

    CLASS_IDX = {}
    CLASS_IDX_INV = {}

    for i, label in enumerate(CLASSES):
        if label != "ambiguous":
            CLASS_IDX[label] = i
            CLASS_IDX_INV[i] = label
        else:
            CLASS_IDX["ambiguous"] = 255
            CLASS_IDX_INV[255] = "ambiguous"
    NUM_CLASS = len(CLASSES) - 1

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self._init_palette()

    def _init_palette(self):
        self.cmap = colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def get_palette(self):
        return self.palette

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image


class WikiSegmentation_corr(WikiScenes_corr):

    corr = None

    def __init__(self, cfg, split, test_mode, root=os.path.expanduser('./data')):
        super(WikiSegmentation_corr, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.test_mode = test_mode

        assert self.split in ['train', 'val', 'val_single', 'val_easy'], "Only support train and val, but get {}".format(self.split)

        # train/val/test splits are pre-cut
        if self.split == 'train':
            _split_f = os.path.join(self.root, 'train_10classes_20639_63landmark_balanced.txt')
        elif self.split == 'val':
            _split_f = os.path.join(self.root, 'val_10classes_20639_7landmark.txt')
        elif self.split == 'val_easy':
            _split_f = os.path.join(self.root, 'val_easy_10classes_20639_63landmark.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        assert os.path.isfile(_split_f), "%s not found" % _split_f

        cache_path = os.path.join(
            self.root, "cache", os.path.splitext(os.path.basename(_split_f))[0] + ".pkl"
        )
        if not os.path.exists(cache_path):
            if not WikiSegmentation_corr.corr:
                with open("correspondence.json", 'r', encoding='utf-8') as f:
                    WikiSegmentation_corr.corr = json.load(f)
            corr = WikiSegmentation_corr.corr
            self.images, self.labels, self.captions, self.tags_list, self.keypoints, self.masks, self.corr_index, self.image_graph = load_dataset(self.root, _split_f, corr)
            if not os.path.exists(os.path.join(self.root, "cache")):
                os.mkdir(os.path.join(self.root, "cache"))
            cPickle.dump([self.images, self.labels, self.captions, self.tags_list, self.keypoints, self.masks, self.corr_index, self.image_graph], open(cache_path, "wb"))
        else:
            print("Loading from %s" % cache_path)
            self.images, self.labels, self.captions, self.tags_list, self.keypoints, self.masks, self.corr_index, self.image_graph = cPickle.load(open(cache_path, "rb"))

        self.transform = tf.Compose([tf.RandResizedCrop_corr(self.cfg.DATASET), \
                                     tf.HFlip_corr(), \
                                     tf.ColourJitter_corr(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), \
                                     tf.Normalise_corr(self.MEAN, self.STD)
                                     ])

        print("{}/{} images have keypoints, {} keypoints in total, {} pairs per image.".format(len(self.image_graph),
                                                                                               len(self.keypoints),
                                                                                               len(self.corr_index),
                                                                                               sum(
                                                                                                   [len(k) for k in
                                                                                                    self.image_graph.values()]) / (
                                                                                                           len(
                                                                                                               self.image_graph) + 0.001)))
        counter = dict()
        for c in classes:
            counter[c] = list()
        for i in self.image_graph:
            for label in self.labels[i]:
                counter[label].append(len(self.image_graph[i]))
        for label in counter:
            print("Class {} has {} paired images, {} pairs per image".format(label, len(counter[label]),
                                                                             sum(counter[label]) / (
                                                                                     len(counter[label]) + 0.001)))
        self.cnt = 0


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        keypoints = self.keypoints[index]
        pair_index = None
        # find a image with correspondence
        if keypoints != None and index in self.image_graph:
            p = random.choice(self.image_graph[index])
            if p != index:
                pair_index = p
        # if no correspondence, randomly pick one
        if pair_index == None:
            pair_index = random.randint(0, len(self.images) - 1)

        images = list()
        label_tensors = list()
        train_labels = list()
        captions = list()
        kp = list()
        for i in [index, pair_index]:
            image = Image.open(self.images[i]).convert('RGB')
            label  = self.labels[i]
            keypoints = self.keypoints[i].copy() if self.keypoints[i] != None else None

            # ignoring BG
            label_tensor = torch.zeros(self.NUM_CLASS - 1)
            train_label = torch.zeros(self.NUM_CLASS - 1) 
            for l in label:
                label_index = self.CLASS_IDX[l]
                label_index -= 1 # shifting since no BG class
                label_tensor[label_index] = 1
            if label:
                train_label[self.CLASS_IDX[label[0]] - 1] = 1

            # general resize, normalize and toTensor
            image, keypoints = self.transform(image, keypoints)
            caption = "captions: {}, tags: {}".format(self.captions[i], self.tags_list[i])

            images.append(image)
            label_tensors.append(label_tensor)
            train_labels.append(train_label)
            captions.append(caption)
            kp.append(keypoints)
        # get correspondence
        common = set(kp[0]).intersection(set(kp[1])) if kp[0] != None and kp[1] != None else None
        if common:
            corr = dict()
            for i in common:
                corr[i] = [*kp[0][i], *kp[1][i]]
            corr = json.dumps(corr)
        else:
            corr = "{}"
        images = {"1": images[0], "2": images[1], "corr": corr, "path": self.images[index], 
            "no. of kp": len(self.keypoints[index]) if self.keypoints[index] != None else 0, 
            "no. of pair": len(self.image_graph[index]) if index in self.image_graph else 0}
        return images, label_tensors, train_labels, captions

    @property
    def pred_offset(self):
        return 0
