# WikiScenes Dataset

| <img src="figures/teaser.PNG" alt="drawing" width="640"/><br> |
|:---|
| This is the official repository for *WikiScenes*, which a large-scale dataset of landmark photo collections that also contains descriptive text in the form of captions and hierarchical category names. <br> You can find the download link for this dataset and also the code to reproduce the result in our paper. |

### The dataset
1. **Image and Caption** WikiScenes contains 63K images with caption. Download the data from:
   - WikiScenes: [ (1.9GB .zip file)](https://drive.google.com/file/d/1w1vlMuW3QrouyMCPZOk8EUrSr8wan74k/view?usp=sharing)

   **Data Structure**
    WikiScenes is organized recursively, following the tree structure in Wikimedia. 
    Each semantic category (e.g. cathedral) contains the following recursive structure:
    ```
    ----0 (e.g., "milano cathedral duomo milan milano italy italia")
    --------0 (e.g., "Exterior of the Duomo (Milan)")
    ----------------0 (e.g., "Duomo (Milan) in art - exterior")
    ----------------1
    ----------------...
    ----------------K0-0
    ----------------category.json
    ----------------pictures (contains all pictures in current hierarchy level)
    --------1
    --------...
    --------K0
    --------category.json
    --------pictures (contains all pictures in current hierarchy level)
    ----1
    ----2
    ----...
    ----N
    ----category.json
    ```
    category.json is a dictionary of the following format: 
    ```
    {
        "max_index": SUB-DIR-NUMBER
        "pairs" :    {
                        CATEGORY-NAME: SUB-DIR-NAME
                    }
        "pictures" : {
                        PICTURE-NAME: {
                                            "caption": CAPTION-DATA,
                                            "url": URL-DATA,
                                            "properties": PROPERTIES
                                    }
                    }
    }
    ```
    where:
    1. SUB-DIR-NUMBER is the total number of subcategories
    2. CATEGORY-NAME is the name of the category (e.g., "milano cathedral duomo milan milano italy italia")
    3. SUB-DIR-NAME is the name of the sub-folder (e.g., "0") 
    4. PICTURE-NAME is the name of the jpg file located within the pictures folder
    5. CAPTION-DATA contains the caption and URL contains the url from which the image was scraped.
    6. PROPERTIES is a list of properties pre-computed for the image-caption pair.
2. **Keypoint correspondences**
   We also provide keypoint correspondences between pixels of images from the same landmark. 
   - correspondence: [ (913MB .zip file)](https://drive.google.com/file/d/1-G2xnrC6RvSnNO9PVmW6w2NGofKV6kW7/view?usp=sharing)

   **Data Structure**
   ```
    {
        "image_id" : {
                        "kp_id": (x, y),
                    }
    }
    ```
    where:
    1. image_id is the id of each image.
    2. kp_id is the id of keypoints, which is identical in the whole dataset.
    3. (x, y) the location of the keypoint in this image.

### Reproduce
1. **Minimum requirements.** This project was originally developed with Python 3.6, PyTorch 1.0 and CUDA 9.0. The training requires at least one Titan X GPU (12Gb memory) .
2. **Setup your Python environment.** Please, clone the repository and install the dependencies:
    ```
    conda create -n <environment_name> --file requirements.txt -c conda-forge/label/cf202003
    conda install scikit-learn=0.21
    pip install opencv-python
    ```
3. **Download and link to the dataset.** First download the data as illustrated above, unzip them.

    Link to the data:
    ```
    ln -s <your_path_to_Wikiscenes> <project>/data/
    ln -s <your_path_to_correspondense.json> <project>/
    ```

4. **Download pre-trained models.** Download the initial weights (pre-trained on ImageNet) for the backbones you are planning to use and place them into `<project>/models/weights/`.

    | Backbone | Initial Weights | Comment |
    |:---:|:---:|:---:|
    | ResNet50 | [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) | PyTorch official |
5. **Train on WikiScenes dataset** 
    
    The first run always takes longer for pre-processing. Computation is cached after then.


### Training, Inference and Evaluation
The directory `launch` contains template bash scripts for training, inference and evaluation. 

**Training.** For each run, you need to specify names of two variables, `bash EXP` and `bash RUN_ID`. 
Running `bash EXP=wiki RUN_ID=v01 ./launch/run_wikiscenes_resnet50.sh` will create a directory `./logs/wikiscenes_corr/wiki/` with tensorboard events and will save snapshots into `./snapshots/wikiscenes_corr/wiki/v01`.

**Inference.** To generate final masks, please, use the script `./launch/infer_val_wikiscenes.sh`. You will need to specify:
* `EXP` and `RUN_ID` you used for training;
* `OUTPUT_DIR` the path where to save the masks;
* `SNAPSHOT` specifies the model suffix in the format `e000Xs0.000`;

**Evaluation.** To compute IoU of the masks, please, run `./launch/eval_seg.sh`. You will need to specify `SAVE_DIR` that contains the masks.

Before running the script, please download our [validation set](https://drive.google.com/file/d/1LS8tsaT6JvbRL3tdYCZcL7MT0ESwinyr/view?usp=sharing)

### Pre-trained model
For testing, we provide our pre-trained ResNet50 model:

| Backbone | Link |
|:---:|---:|
| ResNet50 | [model_enc_e030Xs-0.825.pth (157M)](https://drive.google.com/file/d/1OS1BsO6I7xBBUJlE4uSE-bfZCq6UpA9y/view?usp=sharing) |

## Citation
<!-- We hope that you find this work useful. If you would like to acknowledge us, please, use the following citation:
```
@inproceedings{Araslanov:2020:WSEG,
  title     = {Single-Stage Semantic Segmentation from Image Labels},
  author    = {Araslanov, Nikita and and Roth, Stefan},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
} -->
```
