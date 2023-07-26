import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
import transforms3d
from pycocotools.coco import COCO
from config import cfg
# from utils.renderer import Renderer
import lmdb
from utils.smpl import SMPL
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation, load_img_from_lmdb
from utils.transforms import cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db, denorm_joints, convert_crop_cam_to_orig_img
# from utils.vis import save_obj, vis_keypoints_with_skeleton, vis_bbox, render_mesh

### ONLY FOR TEST
def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord

class OH3D(torch.utils.data.Dataset):
    def __init__(self):
        self.conf_thr = 0.05
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'OH3D', 'J_regressor_h36m_correct.npy'))
        self.datalist = self.load_data()
        print("3doh data len: ", len(self.datalist))
    
    def load_data(self):
        datalist = []
        annots_path = osp.join(cfg.root_dir, 'data', 'OH3D', '3doh_test_annots_scale_1.json') # 在这里加东西，加 pred 2D pose
        annots = json.load(open(annots_path, 'r'))
        for img_path, annot in annots.items():
            annot['img_path'] = img_path
            datalist.append(annot)
        return datalist
    
    def get_mesh(self, annot):
        pose = torch.tensor(annot['pose']).reshape(1, 72).float()
        shape = torch.tensor(annot['shape']).reshape(1, 10).float()
        trans = torch.tensor(annot['trans']).reshape(1, 3).float()
        scale = torch.tensor(annot['scale']).reshape(-1).float()
        mesh, _ = self.smpl.layer['neutral'](pose, shape, trans)
        return mesh.numpy().astype(np.float32).reshape(-1,3)
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        annot = self.datalist[idx]
        img_path = osp.join(cfg.root_dir, 'data', 'OH3D', 'images', annot['img_path'])
        img = cv2.imread(img_path)
        img = img[:, :, ::-1].copy()
        height, weight, _ = img.shape
        
        coco_joint_img = np.array(annot['hrnet_results']).reshape(17, 3)
        joint_coord_img = add_pelvis(coco_joint_img, self.coco_joints_name)
        joint_coord_img = add_neck(joint_coord_img, self.coco_joints_name)
        joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
        joint_valid = np.ones_like(joint_coord_img[:, :1], dtype=np.float32)
        
        joint_valid[joint_coord_img[:, 2] <= self.conf_thr] = 0
        if coco_joint_img.sum() == 0:
            bbox = np.array(annot['bbox']).astype(np.float32).reshape(-1) # xyxy
            # from torchvision.utils import save_image, draw_bounding_boxes, draw_keypoints
            # th_img = torch.from_numpy(img).to(torch.uint8).permute(2, 0, 1).contiguous()
            # th_img = draw_bounding_boxes(th_img, torch.tensor(bbox).reshape(-1, 4), width=5, colors='green')
            # save_image(th_img.float().div(255), './debug/vis_3doh/no_kp_{}.jpg'.format(idx))
            # bbox[2:] = bbox[:2] - bbox[2:] # xywh
            
        else:
            bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
        bbox = process_bbox(bbox.copy(), weight, height, is_3dpw_test=True)

        img, img2bb_trans, bb2img_trans, _, _ = augmentation(img, bbox, 'test')
        img = torch.tensor(img).permute(2,0,1).float().div(255)

        joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
        joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_trunc = joint_valid * (
                    (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.output_hm_shape[2]) * \
                    (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)
        smpl_mesh_cam = self.get_mesh(annot)
        inputs = {'img': img, 'joints': joint_coord_img, 'joints_mask': joint_trunc}
        targets = {'smpl_mesh_cam': smpl_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans, 'bbox': bbox, 'img_path': img_path, 'idx': idx}
       
        return inputs, targets, meta_info
    
    def random_idx_eval(self, outs, index):
        annots = self.datalist
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
        for out, idx in zip(outs, index):
            # annot = annots[idx]
            mesh_gt_cam = out['smpl_mesh_cam_target']
            pose_coord_gt_h36m = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            # debug
            # root_h36m_gt = pose_coord_gt_h36m[self.h36m_root_joint_idx, :]
            # pose_gt_img = cam2pixel(pose_coord_gt_h36m, annot['cam_param']['focal'], annot['cam_param']['princpt'])
            # pose_gt_img = transform_joint_to_other_db(pose_gt_img, self.h36m_joints_name, self.smpl.graph_joints_name)

            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint, :]
            mesh_gt_cam -= np.dot(self.joint_regressor, mesh_gt_cam)[0, None, :]

            mesh_out_cam = out['smpl_mesh_cam']
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)

            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint, :]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)

            eval_result['mpjpe'].append(np.sqrt(
                np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m) ** 2,
                                                          1)).mean() * 1000)  # meter -> milimeter
            mesh_out_cam -= np.dot(self.joint_regressor, mesh_out_cam)[0, None, :]

            mesh_error = np.sqrt(np.sum((mesh_gt_cam - mesh_out_cam) ** 2, 1)).mean() * 1000
            eval_result['mpvpe'].append(mesh_error)
        return eval_result
    
    def evaluate(self, outs, cur_sample_idx):
        index = [None] * len(outs)
        return self.random_idx_eval(outs, index)
