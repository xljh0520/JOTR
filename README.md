<p align="center">

  <h1 align="center">JOTR: 3D Joint Contrastive Learning with Transformers for Occluded Human Mesh Recovery 
    
  </h1>
  <h2 align="center">ICCV 2023</h2>
 <div align="center">Official PyTorch implementation of the paper "JOTR: 3D Joint Contrastive Learning with Transformers for Occluded Human Mesh Recover" </div>
 <div align="center">
  </div>
</p>
<p float="center">
  <div align="center">
  </div>
</p>

[[Preprint](https://arxiv.org/abs/2307.16377)]

![demo_0](demo/demo_0.gif)

![demo_1](demo/demo_1.gif)

![demo_2](demo/demo_2.gif)

<!-- ### Code is coming soon -->



### Installation
```
conda create python=3.7 --name jotr
conda activate jotr
pip install torch==1.8.0 torchvision==0.9.0
sh requirements.sh
```
<!-- pip install -r requirements.txt -->


### Data Preparation

We prepare the data in a samilar way as [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md). Please refer to [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE/blob/main/assets/directory.md) for dataset, SMPL model, VPoser model, and backbone pre-trained weights.

Download the annotations of [3DPW-PC](https://drive.google.com/file/d/1xzZvUj1lR1ECbzUI4JOooC_r2LF6Qs5m/view?usp=sharing), [3DPW-OC](https://drive.google.com/file/d/1IPE8Yw7ysd97Uv6Uw24el1yRs2r_HtCR/view?usp=sharing).

Download checkpoint of JOTR from [here](https://drive.google.com/drive/folders/1kJe34jWKlQh14M4reptgdnXVg8Egle19?usp=sharing).

The data directory should be organized as follows:

```  
${ROOT}  
|-- checkpoint
|-- 3dpw_best_ckpt.pth.tar
|-- 3dpw-crowd_best_ckpt.pth.tar
|-- 3dpw-oc_best_ckpt.pth.tar
|-- 3dpw-pc_best_ckpt.pth.tar
|-- data 
|   |-- J_regressor_extra.npy 
|   |-- snapshot_0.pth.tar
|   |-- 3DPW
|   |   |-- 3DPW_latest_test.json
|   |   |-- 3DPW_oc.json
|   |   |-- 3DPW_pc.json
|   |   |-- 3DPW_validation_crowd_hhrnet_result.json
|   |   |-- imageFiles
|   |   |-- sequenceFiles
|   |-- CrowdPose
|   |   |-- annotations
|   |   |-- images
|   |-- Human36M  
|   |   |-- images  
|   |   |-- annotations   
|   |   |-- J_regressor_h36m_correct.npy
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |-- annotations  
|   |   |-- J_regressor_coco_hip_smpl.npy
|   |-- MuCo  
|   |   |-- augmented_set  
|   |   |-- unaugmented_set  
|   |   |-- MuCo-3DHP.json
|   |   |-- smpl_param.json

```  

### Evaluation
Reproduce the results in the paper (Table 1 and Table 2) by running the following command:
```
sh eval.sh
```

### Training
Train the model by running the following command:

```
sh train.sh
```
### Visualization
TODO

<!-- ### Citation
```

``` -->

## Acknowledgments

Thanks to [3DCrowdNet](https://github.com/hongsukchoi/3DCrowdNet_RELEASE), [DETR](https://github.com/facebookresearch/detr), [AutomaticWeightedLoss](https://github.com/Mikoto10032/AutomaticWeightedLoss), [deep_training](https://github.com/ssbuild/deep_training) and [PositionalEncoding2D](https://github.com/wzlxjtu/PositionalEncoding2Dl), our code is partially borrowing from them.


## License

This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including SMPL, VPoser, and uses datasets that have their own licenses. Please refer to the corresponding websites for more details.

## Citation
```
@article{li2023jotr,
  title={JOTR: 3D Joint Contrastive Learning with Transformers for Occluded Human Mesh Recovery},
  author={Li, Jiahao and Yang, Zongxin and Wang, Xiaohan and Ma, Jianxin and Zhou, Chang and Yang, Yi},
  journal={ICCV},
  year={2023}
}
```
