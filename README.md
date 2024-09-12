# SuRF
SuRF is the first unsupervised method to achieve end-to-end sparsification, to the best of our knowledge. It can utilize the higher-resolution volume to reconstruct higher-frequency geometric details with less computation and memory consumption. Details are described in our paper:
> Surface-Centric Modeling for High-Fidelity Generalizable Neural Surface Reconstruction
>
> Rui Peng, Shihe Shen, Kaiqiang Xiong, Huachen Gao, Jianbo Jiao, Xiaodong Gu, Ronggang Wang
>
> ECCV 2024 ([arxiv](https://arxiv.org/abs/2409.03634))

<p align="center">
    <img src="./.github/images/sample.gif" width="100%"/>
</p>

📍 If there are any bugs in our code, please feel free to raise your issues.

## ⚙ Setup
#### 1. Recommended environment
```
conda create -n surf python=3.10.9
conda activate surf
pip install -r requirements.txt
```

#### 2. DTU Dataset

We only train our model on DTU dataset. We adopt the full resolution ground-truth depth maps (just for testing) and RGB images, and use the camera parameters prepocessed by CasMVSNet or MVSNet. Simply, please follow the instruction [here](https://github.com/prstrive/UniMVSNet/tree/main#2-dtu-dataset) of UniMVSNet to prepare the dataset. We generate [pseudo points]() and [pseudo depths](https://drive.google.com/file/d/1LYxsH345zqAVIPcy67mSZl_4asB8thAZ/view?usp=sharing) through RC-MVSNet to assist the model optimization, download and unzip them. The final data structure is just like this:
```
dtu_training                          
  ├── Cameras
    ├── 00000000_cam.txt
    ├── ...             
    ├── pair.txt
  ├── Depths_raw
  ├── Pseudo_depths
  ├── Pseudo_points
  └── Rectified_raw
```
`Rectified_raw` is the full resolution RGB images provided in [DTU](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip). We use the same training and testing split as SparseNeuS, please refer to [here](datasets/dtu_split) for more details.

For testing, you can download the testing data prepared by SparseNeuS [here](https://connecthkuhk-my.sharepoint.com/personal/xxlong_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxxlong%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsparseneus%2FDTU%5FTEST%2Ezip&parent=%2Fpersonal%2Fxxlong%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsparseneus&ga=1), which contains some object masks for cleaning the mesh. Put it to `<your DTU_TEST path>`. For quantitative evaluation, you need to download the ground-truth points from the [DTU website](https://roboimagedata.compute.dtu.dk/?page_id=36) and put it to `<your GT_POINTS path>`.

#### 3. BlendedMVS Dataset

Download [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) 
for evaluation. The data structure is just like this:
```
blendedmvs                          
  ├── 5a0271884e62597cdee0d0eb
    ├── blended_images
    ├── cams
      ├── 00000000_cam.txt
      ├── ...
      └── pair.txt  
  ├── 5a3ca9cb270f0e3f14d0eddb
  ├── ...
```

#### 4. Tanks and Temples Dataset

Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) 
for evaluation. The data structure is just like this:
```
tanksandtemples                          
  ├── advanced
    ├── Auditorium
    └── ...
  ├── intermediate
    ├── Family
    └── ...
```

#### 5. ETH3D

Download [ETH3D](https://www.eth3d.net/) 
for evaluation following their official instructions. The data structure is just like this:
```
ETH3D                          
  ├── courtyard
    ├── cams
    └── images
  ├── ...
```

## 📊 Testing

#### 1. Download models or precomputed meshes
Download our pretrained model and put it to `<your CKPT path>`.
<table align="center">
  <tr align="center">
    <td>CKPT</td>
    <td>Meshes</td>
    <td>Train Res</td>
		<td>Train View</td>
		<td>Test Res</td>
		<td>Test View</td>
		<td>Mean Cham. Dist.↓</td>
	</tr>
	<tr align="center">
    <td><a href="https://drive.google.com/file/d/1EKdrcoEW1pedeVkQv69G1gt2f1uLDMSh/view?usp=sharing">surf</a></td>
    <td><a href="https://drive.google.com/drive/folders/1bAvrvaq143GToIIYCaxDnoyr6lKC6wiM?usp=sharing">meshes</a></td>
		<td>480X640</td>
		<td>5 (4src)</td>
		<td>576X800</td>
		<td>3 (2src)</td>
		<td>1.05</td>
	</tr>
</table>

You can also download our precomputed DTU meshes through direct inference.

#### 2. DTU testing

**Mesh Extraction.** We define all information like the model structure and testing parameters in the configuration file. We use the `./confs/surf.conf` file for training and testing. You need to first specify the correct values in your own environment, such as `<your dataset path>` and `<your output save path>`. You can use our default testing configurations and the model structure. Once everything is ready, you can simply start testing via:
```
bash ./scripts/run.sh --mode val --resume <your CKPT path>
```
This will predict all scenes in the [test split](/home/pengr/Documents/GenS/datasets/dtu_split/test.txt) at view index 23 by default. If you want to get the results at other views (e.g., view43), you can change the `ref_view` under the `val_dataset` namespace in configuration file. Meanwhile, you can also specify `scene` list under the `val_dataset` namespace to test on a single scene like `scene=[scan24,scan55]`.

Optionaly, you can add `--clean_mesh` command to generate the filtered mesh, but you need to note that the mask used in `--clean_mesh` command is from MVSNet and is not the correct object mask used during the [quantitative evaluation](evaluation/clean_meshes.py).

**Mesh cleaning.** Before evaluation, to generate the clean meshes using the correct mask, you need clean the mesh first:
```
python evaluation/clean_meshes.py --root_dir <your DTU_TEST path> --out_dir <your output save path>/meshes
```

**Quantitative Evaluation.** Run:
```
python evaluation/dtu_eval.py --dataset_dir <your GT_POINTS path> --out_dir <your output save path>
```

You need to pay attention to the filename of meshes in `evaluation/clean_meshes.py` file.

#### 3. BlendedMVS testing

Using the configuration file `confs/gens_bmvs.conf` to evaluate on BlendedMVS dataset, run:
```
python main.py --conf confs/gens_bmvs.conf --mode val --resume <your CKPT path> --clean_mesh
```
Here, we recommand to add the `--clean_mesh` command. You can change or add more testing scenes through change `scene`. Note that camera poses in BlendedMVS have a great difference, you need to make sure that the bounding box fits as closely as possible to the object you want to reconstruct, e.g., adjusting `factor` and `num_interval`.

#### 4. Tanks and Temples testing

Using the configuration file `confs/gens_tanks.conf` to evaluate on Tanks and Temples dataset, run:
```
python main.py --conf confs/gens_tanks.conf --mode val --resume <your CKPT path> --clean_mesh
```

#### 5. ETH3D testing

Using the configuration file `confs/gens_eth3d.conf` to evaluate on ETH3D dataset, run:
```
python main.py --conf confs/gens_eth3d.conf --mode val --resume <your CKPT path> --clean_mesh
```

## ⏳ Training & Fine-tuning

#### 1. DTU training

During training, we set the lowest resolution of volume to 64, you need to first specify the value in `confs/surf.conf` file and run:
```
bash ./scripts/run.sh --mode train
```
By default, we employ the *DistributedDataParallel* mode to train our model on 2 GPUs.

#### 2. DTU fine-tuning

We use `confs/surf_finetune.conf` file to config the fine-tuning on DTU dataset. For convenience, we use `scripts/finetune.sh` file to fine-tune all testing scenes at both 23 and 43 views:
```
bash ./scripts/finetune.sh --resume <your CKPT path>
```

You can change the scene and view through the `--scene` and `--ref_view` command directly or through modifying the configuration file. 

*You can fine-tune or train on other datasets in a similar manner.*

## ⚖ Citation
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{peng2024eccv,
  title={Surface-Centric Modeling for High-Fidelity Generalizable Neural Surface Reconstruction},
  author={Peng, Rui and Shihe, Shen and Xiong, Kaiqiang and Gao, Huachen and Jiao, Jianbo and Gu, Xiaodong and Wang, Ronggang},
  booktitle={The 18th European Conference on Computer Vision (ECCV)},
  year={2024}
}

```

## 👩‍ Acknowledgements

This code is highly inhreated from [GenS](https://github.com/prstrive/GenS), and we use the SDF-based volume rendering in [NeuS](https://github.com/Totoro97/NeuS) and blending in [IBRNet](https://github.com/googleinterns/IBRNet). Thanks for their great contributions!

