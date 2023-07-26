import os
import os.path as osp
import sys
import numpy as np
import datetime
import yaml
import shutil
import glob
from easydict import EasyDict as edict
import torch.distributed as dist

class Config:
    ## dataset
    # MuCo, Human36M, MSCOCO, PW3D, FreiHAND
    trainset_3d = ['Human36M']  # 'Human36M', 'MuCo'
    trainset_2d = ['MSCOCO']  # 'MSCOCO', 'MPII', 'CrowdPose'
    testset = 'PW3D'  # 'MuPoTs' 'MSCOCO' Human36M, MSCOCO, 'PW3D'

    ## model setting
    resnet_type = 50  # 50, 101, 152
    frozen_bn = False
    distributed = False
    upsample_net = False
    use_cls_token = False # if True use cls token else mean pooling
    num_layers = 6
    enc_layers = 3
    dec_layers = 3
    local_rank = 0
    max_norm = 0
    weight_decay = 0
    is_local = False
    resume_ckpt = ''
    update_bbox = False

    ## input, output
    input_img_shape = (256, 256)  #(256, 192)
    output_hm_shape = (64, 64, 64)  #(64, 64, 48)
    bbox_3d_size = 2 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 0.3
    sigma = 2.5
    focal = (5000, 5000)  # virtual focal lengths
    princpt = (input_img_shape[1] / 2, input_img_shape[0] / 2)  # virtual principal point position

    ## training config
    lr_dec_epoch = [15] if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else [17,21]
    end_epoch = 20 #13 if 'FreiHAND' not in trainset_3d + trainset_2d + [testset] else 25
    lr = 1e-4
    lr_backbone = 1e-4
    lr_dec_factor = 10
    train_batch_size = 64
    use_gt_info = True

    ## testing config
    test_batch_size = 64
    crowd = False
    vis = False
    render = False

    ## others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    bbox_3d_size = 2
    camera_3d_size = 2.5
    
    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir)
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')

    save_folder = 'JOTR'
    output_dir = osp.join(output_dir, save_folder)
    print('output dir: ', output_dir)

    model_dir = osp.join(output_dir, 'checkpoint')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
    mano_path = osp.join(root_dir, 'common', 'utils', 'manopth')
    smpl_path = osp.join(root_dir, 'common', 'utils', 'smplpytorch')
    human_model_path = osp.join(root_dir, 'common', 'utils', 'human_model_files')

    def set_args(self, continue_train=False, is_test=False, resume_ckpt=''):
        if not is_test:
            self.continue_train = continue_train
            if self.continue_train:
                if resume_ckpt and osp.exists(resume_ckpt):
                    shutil.copy(resume_ckpt, cfg.model_dir)

                else:
                    shutil.copy(osp.join(cfg.root_dir, 'data', 'snapshot_0.pth.tar'), osp.join(cfg.model_dir, 'snapshot_0.pth.tar'))
                
        if self.testset == 'FreiHAND':
            assert self.trainset_3d[0] == 'FreiHAND'
            assert len(self.trainset_3d) == 1
            assert len(self.trainset_2d) == 0

    def update(self, args, folder_dict=None):
        for arg in vars(args):
            setattr(cfg, arg, getattr(args, arg))

        with open(args.cfg) as f:
            exp_config = edict(yaml.load(f))
            for k, v in exp_config.items():
                if hasattr(args, k):
                    v = getattr(args, k)
                setattr(cfg, k, v)
               
        if folder_dict is not None:
            for k, v in folder_dict.items():
                setattr(cfg, k, v)
        return exp_config


cfg = Config()
sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
dataset_list = ['CrowdPose', 'Human36M', 'MPII', 'MSCOCO', 'MuCo', 'PW3D']
for i in range(len(dataset_list)):
    add_pypath(osp.join(cfg.data_dir, dataset_list[i]))
