# Towers of Babel: Combining Images, Language, and 3D Geometry for Learning Multimodal Vision
Download links and PyTorch implementation of "Towers of Babel: Combining Images, Language, and 3D Geometry for Learning Multimodal Vision", ICCV 2021.
> Towers of Babel: Combining Images, Language, and 3D Geometry for Learning Multimodal Vision
> [Xiaoshi Wu](https://github.com/tgxs002), [Hadar Averbuch-Elor](cs.cornell.edu/~hadarelor), [Jin Sun](cs.cornell.edu/~jinsun), [Noah Snavely](https://www.cs.cornell.edu/~snavely/) 
> ICCV 2021
>
#### [Project Page](https://www.cs.cornell.edu/projects/babel/) | [Paper](https://www.cs.cornell.edu/projects/babel/) 

| <img src="figures/teaser.PNG" alt="drawing" width="800"/><br> |
|:---|

### The WikiScenes Dataset
1. **Image and Textual Descriptions:** WikiScenes contains 63K images with captions. We provide two versions for download:
   - Low-res version used in our experiments (shorter dimension set to 200[px], aspect ratio fixed): [ (1.9GB .zip file)](https://www.cs.cornell.edu/projects/babel/WikiScenes.zip)
   - Higher-res version (longer dimension set to 1200[px], aspect ratio fixed): [ (19.4GB .zip file)](https://www.cs.cornell.edu/projects/babel/WikiScenes1200px.zip)

   *Data Structure*
   
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
    6. PROPERTIES is a list of properties pre-computed for the image-caption pair (e.g. estimated language of caption).
2. **Keypoint correspondences:** We also provide keypoint correspondences between pixels of images from the same landmark: [ (982MB .zip file)](https://www.cs.cornell.edu/projects/babel/correspondence.json.zip)

   *Data Structure*
   ```
    {
        "image_id" : {
                        "kp_id": (x, y),
                    }
    }
    ```
    where:
    1. image_id is the id of each image.
    2. kp_id is the id of keypoints, which is unique across the whole dataset.
    3. (x, y) the location of the keypoint in this image.
3. **COLMAP reconstructions:** We provide the full 3D models used for computing keypoint correspondences: [ (1GB .zip file)](https://www.cs.cornell.edu/projects/babel/WikiScenes3D.zip)

### Reproducing Results
1. **Minimum requirements.** This project was originally developed with Python 3.6, PyTorch 1.0 and CUDA 9.0. The training requires at least one Titan X GPU (12Gb memory) .
2. **Setup your Python environment.** Clone the repository and install the dependencies:
    ```
    conda create -n <environment_name> --file requirements.txt -c conda-forge/label/cf202003
    conda install scikit-learn=0.21
    pip install opencv-python
    ```
3. **Download the dataset.** Download the data as detailed above, unzip and place as follows: Image and textual descriptions in ```<project>/data/``` and the correspondence file in ```<project>```.

4. **Download pre-trained models.** Download the initial weights (pre-trained on ImageNet) for the backbone model and place in `<project>/models/weights/`.

    | Backbone | Initial Weights | Comments |
    |:---:|:---:|:---:|
    | ResNet50 | [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) | PyTorch official model|
5. **Train on the WikiScenes dataset.** See instructions below. Note that the first run always takes longer for pre-processing. Some computations are cached afterwards.


### Training, Inference and Evaluation
The directory `launch` contains template bash scripts for training, inference and evaluation. 

**Training.** For each run, you need to specify the names of two variables, `bash EXP` and `bash RUN_ID`. 
Running `bash EXP=wiki RUN_ID=v01 ./launch/run_wikiscenes_resnet50.sh` will create a directory `./logs/wikiscenes_corr/wiki/` with tensorboard events and saved snapshots in `./snapshots/wikiscenes_corr/wiki/v01`.

**Inference.** To generate final masks, run `./launch/infer_val_wikiscenes.sh`. You will need to specify:
* `EXP` and `RUN_ID` you used for training;
* `OUTPUT_DIR` the path where to save the masks;
* `SNAPSHOT` specifies the model suffix in the format `e000Xs0.000`;

**Evaluation.** To compute IoU of the masks, run `./launch/eval_seg.sh`. You will need to specify `SAVE_DIR` that contains the masks.

Before running the script, please download our [validation set](https://drive.google.com/file/d/1LS8tsaT6JvbRL3tdYCZcL7MT0ESwinyr/view?usp=sharing).

### Pre-trained model
For testing, we provide our pre-trained ResNet50 model:

| Backbone | Link |
|:---:|---:|
| ResNet50 | [model_enc_e030Xs-0.825.pth (157M)](https://drive.google.com/file/d/1OS1BsO6I7xBBUJlE4uSE-bfZCq6UpA9y/view?usp=sharing) |

## Citation
```
@inproceedings{Wu2021Towers,
 title={Towers of Babel: Combining Images, Language, and 3D Geometry for Learning Multimodal Vision},
 author={Wu, Xiaoshi and Averbuch-Elor, Hadar and Sun, Jin and Snavely, Noah},
 booktitle={ICCV},
 year={2021}
}
```

## Acknowledgement
Our code is based on the implementation of [Single-Stage Semantic Segmentation from Image Labels](https://github.com/visinf/1-stage-wseg)