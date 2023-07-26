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


class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_name):
        print('='*20, 'PW3D', '='*20)
        self.transform = transform
        assert data_name in ['3dpw', '3dpw-pc', '3dpw-oc', '3dpw-crowd']
        self.data_name = data_name
        if self.data_name in ['3dpw', '3dpw-pc', '3dpw-oc']:
            self.data_split = 'test'
            cfg.crowd = False
        elif self.data_name == '3dpw-crowd':
            self.data_split = 'validation'
            cfg.crowd = True
        # self.data_split ='validation' if cfg.crowd else 'test'  # data_split
        self.data_path = osp.join(cfg.root_dir, 'data', '3DPW')
        self.human_bbox_root_dir = osp.join(cfg.root_dir, 'data', '3DPW', 'rootnet_output', 'bbox_root_pw3d_output.json')

        # SMPL joint set
        self.smpl = SMPL()
        self.face = self.smpl.face
        self.joint_regressor = self.smpl.joint_regressor
        self.vertex_num = self.smpl.vertex_num
        self.joint_num = self.smpl.joint_num
        self.joints_name = self.smpl.joints_name
        self.skeleton = self.smpl.skeleton
        self.root_joint_idx = self.smpl.root_joint_idx
        self.face_kps_vertex = self.smpl.face_kps_vertex

        # H36M joint set
        self.h36m_joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.h36m_root_joint_idx = self.h36m_joints_name.index('Pelvis')
        self.h36m_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.h36m_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'Human36M', 'J_regressor_h36m_correct.npy'))

        # mscoco skeleton
        self.coco_joint_num = 18+1 # original: 17, manually added pelvis
        self.coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_skeleton = ( (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6), (11, 12) )
        self.coco_flip_pairs = ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) )
        self.coco_joint_regressor = np.load(osp.join(cfg.root_dir, 'data', 'MSCOCO', 'J_regressor_coco_hip_smpl.npy'))

        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Pelvis')
        self.conf_thr = 0.05

        self.datalist = self.load_data()
        # torch.save({
        #     'anno': self.datalist[0:10]
        # }, 'debug/vis_3dpw_gt.pth')
        # import pdb;pdb.set_trace()
        print("3dpw data len: ", len(self.datalist))

    def add_pelvis(self, joint_coord, joints_name):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2]  # confidence for openpose
        pelvis = pelvis.reshape(1, 3)

        joint_coord = np.concatenate((joint_coord, pelvis))

        return joint_coord

    def add_neck(self, joint_coord, joints_name):
        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
        neck = neck.reshape(1,3)

        joint_coord = np.concatenate((joint_coord, neck))

        return joint_coord

    def load_data(self):
        if self.data_name == '3dpw':
            db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))
        elif self.data_name == '3dpw-pc':
            db = COCO(osp.join(self.data_path, '3DPW_pc.json'))
        elif self.data_name == '3dpw-oc':
            db = COCO(osp.join(self.data_path, '3DPW_oc.json'))
        elif self.data_name == '3dpw-crowd':
            db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))

        # db = COCO('/home/ljh375649/data_2/vibe_data/3dpw/3DPW_oc.json')
        # db = COCO('/home/ljh375649/data_2/vibe_data/3dpw/3DPW_pc.json')
        # db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))
        # import pdb;pdb.set_trace()
        if self.data_split == 'test' and not cfg.use_gt_info:
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                ann_id = str(annot[i]['ann_id'])
                bbox_root_result[ann_id] = {'bbox': np.array(annot[i]['bbox']), 'root': np.array(annot[i]['root_cam'])}
        elif cfg.crowd:
            with open(osp.join(self.data_path, f'3DPW_{self.data_split}_crowd_hhrnet_result.json')) as f:
                hhrnet_result = json.load(f)
            print("Load Higher-HRNet input")

        else:
            print("Load OpenPose input")

        hhrnet_count = 0
        datalist = []

        
        for aid in db.anns.keys():
            aid = int(aid)

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']

            if cfg.crowd and self.data_split=='validation':
                if 'courtyard_hug_00' not in sequence_name and 'courtyard_dancing_00' not in sequence_name:
                    continue

            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k,v in img['cam_param'].items()}
            smpl_param = ann['smpl_param']
            if self.data_split == 'test' and not cfg.use_gt_info: # 永远False
                bbox = bbox_root_result[str(aid)]['bbox']  # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(aid)]['root'][2]
            else:
                ann['bbox'] = np.array(ann['bbox'], dtype=np.float32)

                bbox = process_bbox(ann['bbox'], img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = None

            openpose = np.array(ann['openpose_result'], dtype=np.float32).reshape(-1, 3)
            openpose = self.add_pelvis(openpose, self.openpose_joints_name)
            pose_score_thr = self.conf_thr

            hhrnetpose = None
            if cfg.crowd and self.data_split=='validation':
                try:
                    hhrnetpose = np.array(hhrnet_result[str(aid)]['coco_joints'])
                    hhrnetpose = self.add_pelvis(hhrnetpose, self.coco_joints_name)
                    hhrnetpose = self.add_neck(hhrnetpose, self.coco_joints_name)
                    hhrnet_count += 1

                except:
                    hhrnetpose = openpose
                    hhrnetpose = transform_joint_to_other_db(hhrnetpose, self.openpose_joints_name, self.coco_joints_name)
            
            datalist.append({
                'ann_id': aid,
                'img_path': img_path,
                'img_shape': (img_height, img_width),
                'bbox': bbox,
                'tight_bbox': ann['bbox'],
                'smpl_param': smpl_param,
                'cam_param': cam_param,
                'root_joint_depth': root_joint_depth,
                'pose_score_thr': pose_score_thr,
                'openpose': openpose,
                'hhrnetpose': hhrnetpose
            })

        print("check hhrnet input: ", hhrnet_count)
        return datalist

    def get_smpl_coord(self, smpl_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        smpl_pose = torch.FloatTensor(pose).view(1,-1); smpl_shape = torch.FloatTensor(shape).view(1,-1); # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_trans = torch.FloatTensor(trans).view(-1,3) # translation vector from smpl coordinate to 3dpw camera coordinate

        # TEMP
        # gender = 'neutral'
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.smpl.layer[gender](smpl_pose, smpl_shape, smpl_trans)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1,3);
        # smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1,3)
        # smpl_face_kps_coord = smpl_mesh_coord[self.face_kps_vertex,:].reshape(-1,3)
        # smpl_joint_coord = np.concatenate((smpl_joint_coord, smpl_face_kps_coord))
        smpl_joint_coord = np.dot(self.joint_regressor, smpl_mesh_coord)

        return smpl_mesh_coord, smpl_joint_coord

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # if not hasattr(self, 'lmdb'):
        #     db_path = osp.join(cfg.root_dir, 'data/PW3D/data/imageFiles_lmdb/imageFiles_lmdb.lmdb')
        #     env = lmdb.open(db_path,
        #                     subdir=os.path.isdir(db_path),
        #                     readonly=True, lock=False,
        #                     readahead=False, meminit=False)
        #     self.lmdb = env.begin(write=False)
        data = copy.deepcopy(self.datalist[idx])
        aid, img_path, bbox, smpl_param, cam_param = data['ann_id'], data['img_path'], data['bbox'], data['smpl_param'], data['cam_param']
        # import pdb;pdb.set_trace()
        # get gt img joint from smpl coordinates
        smpl_mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_param)
        smpl_coord_img = cam2pixel(smpl_joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_coord_img = smpl_coord_img
        joint_valid = np.ones_like(joint_coord_img[:, :1], dtype=np.float32)

        if cfg.crowd and self.data_split == 'validation':
            # get input joint img from higher hrnet
            joint_coord_img = data['hhrnetpose']
            joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.coco_joints_name, self.joints_name)
        else:
            # get input joint img from openpose
            joint_coord_img = data['openpose']
            joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.openpose_joints_name, self.joints_name)
        pose_thr = data['pose_score_thr']
        joint_valid[joint_coord_img[:, 2] <= pose_thr] = 0

        # get bbox from joints
        bbox = get_bbox(joint_coord_img, joint_valid[:, 0])
        img_height, img_width = data['img_shape']
        bbox = process_bbox(bbox.copy(), img_width, img_height, is_3dpw_test=True)
        bbox = data['bbox'] if bbox is None else bbox

        # img
        img = load_img(img_path)
        # img = load_img_from_lmdb(img_path, self.lmdb)
        """
        # vis
        from torchvision.utils import save_image, draw_bounding_boxes, draw_keypoints
        import torch.nn.functional as F
        th_img = torch.from_numpy(img).to(torch.uint8).permute(2, 0, 1).contiguous()
        processed_bbox = torch.tensor(bbox).float().view(-1, 4)
        processed_bbox[:, 2:] += processed_bbox[:, :2]
        th_img = draw_bounding_boxes(th_img, processed_bbox, width=5, colors='red')
        data_bbox = torch.tensor(get_bbox(joint_coord_img, joint_valid[:, 0])).float().view(-1, 4)
        data_bbox[:, 2:] += data_bbox[:, :2]
        th_img = draw_bounding_boxes(th_img, data_bbox, width=5, colors='green')
        # tight_bbox = torch.tensor(data['tight_bbox']).float().view(-1, 4)
        # tight_bbox[:, 2:] += tight_bbox[:, :2]
        # th_img = draw_bounding_boxes(th_img, tight_bbox, width=5, colors='blue')
            
        vis_joints = torch.tensor(joint_coord_img).float().view(1, -1, 3)
        th_img = draw_keypoints(th_img, vis_joints, width=10, colors='yellow')

        th_img = F.interpolate(th_img[None], scale_factor=0.25)[0]
        save_image(th_img.float().div(255), './debug/vis_3dpw_oc/3dpw_oc_{}.jpg'.format(idx))
        """

        img, img2bb_trans, bb2img_trans, _, _ = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.

        """
        # vis
        joint_coord_img = transform_joint_to_other_db(joint_coord_img, self.joints_name, self.crowdpose_joints_name)
        img = cv2.imread(img_path)
        input_img = vis_keypoints_with_skeleton(img, joint_coord_img.T, self.crowdpose_skeleton, kp_thresh=self.openpose_thr, alpha=1, kps_scores=joint_coord_img[:,2:])
        cv2.imshow('open pose', input_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        import pdb; pdb.set_trace()
        
        # smpl_coord_img[:, 2] = 1
        # input_img = vis_keypoints_with_skeleton(img_copy, smpl_coord_img.T, self.skeleton, kp_thresh=0.4, alpha=1)
        # cv2.imshow('smpl gt', input_img/255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        """

        # x,y affine transform, root-relative depth
        joint_coord_img_xy1 = np.concatenate((joint_coord_img[:, :2], np.ones_like(joint_coord_img[:, 0:1])), 1)
        joint_coord_img[:, :2] = np.dot(img2bb_trans, joint_coord_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
        joint_coord_img[:, 0] = joint_coord_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        joint_coord_img[:, 1] = joint_coord_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        # check truncation
        joint_trunc = joint_valid * (
                    (joint_coord_img[:, 0] >= 0) * (joint_coord_img[:, 0] < cfg.output_hm_shape[2]) * \
                    (joint_coord_img[:, 1] >= 0) * (joint_coord_img[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)

        """
        print(f'{img_path} trunc:\n', joint_trunc.nonzero())
        tmp_coord = joint_coord_img[:, :2] * np.array([[cfg.input_img_shape[1] / cfg.output_hm_shape[2], cfg.input_img_shape[0]/ cfg.output_hm_shape[1]]])
        newimg = vis_keypoints(img.numpy().transpose(1,2,0), tmp_coord)
        cv2.imshow(f'{img_path}', newimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        """

        # from torchvision.utils import save_image
        # print(img_path)
        # save_image(img, 'debug/tmp_image.png')
        # import pdb;pdb.set_trace()

        inputs = {'img': img, 'joints': joint_coord_img, 'joints_mask': joint_trunc}
        targets = {'smpl_mesh_cam': smpl_mesh_cam}
        meta_info = {'bb2img_trans': bb2img_trans, 'img2bb_trans': img2bb_trans, 'bbox': bbox, 'tight_bbox': data['tight_bbox'], 'aid': aid, 'idx': idx, 'img_path':img_path}
        return inputs, targets, meta_info

    def random_idx_eval(self, outs, index):
        annots = self.datalist
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
        for out, idx in zip(outs, index):
            annot = annots[idx]
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
            ### add out['joint_img'] for J3D' evaluation
            mid_j3d = out['joint_img'] # 15, 3
            gt_j3d_30 = np.dot(self.joint_regressor, mesh_gt_cam)
            gt_j3d_15 = gt_j3d_30[self.smpl.idx_list_15]
            gt_j3d_15 = gt_j3d_15 - gt_j3d_15[self.smpl.root_joint_idx, None]  # root-relative
            mid_j3d = mid_j3d - mid_j3d[self.smpl.root_joint_idx, None]  # root-relative
            ### add out['joint_img'] for J3D' evaluation

            ### J3D evaluation
            final_j3d_30 = np.dot(self.joint_regressor, mesh_out_cam)
            final_j3d_15 = final_j3d_30[self.smpl.idx_list_15]
            final_j3d_15 = final_j3d_15 - final_j3d_15[self.smpl.root_joint_idx, None]  # root-relative
            ### J3D evaluation
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint, :]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)

            eval_result['mpjpe'].append(np.sqrt(
                np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m) ** 2,
            #                                               1)).mean() * 1000)  # meter -> milimeter
            # eval_result['mpjpe'].append(np.sqrt(
            #     np.sum((mid_j3d - gt_j3d_15) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            # eval_result['pa_mpjpe'].append(np.sqrt(np.sum((final_j3d_15 - gt_j3d_15) ** 2,
                                                          1)).mean() * 1000)  # meter -> milimeter
            mesh_out_cam -= np.dot(self.joint_regressor, mesh_out_cam)[0, None, :]

            mesh_error = np.sqrt(np.sum((mesh_gt_cam - mesh_out_cam) ** 2, 1)).mean() * 1000
            eval_result['mpvpe'].append(mesh_error)

        return eval_result

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
        pred_list = []
        gt_list = []
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            # h36m joint from gt mesh
            mesh_gt_cam = out['smpl_mesh_cam_target']
            pose_coord_gt_h36m = np.dot(self.h36m_joint_regressor, mesh_gt_cam)
            # debug
            root_h36m_gt = pose_coord_gt_h36m[self.h36m_root_joint_idx, :]
            pose_gt_img = cam2pixel(pose_coord_gt_h36m, annot['cam_param']['focal'], annot['cam_param']['princpt'])
            pose_gt_img = transform_joint_to_other_db(pose_gt_img, self.h36m_joints_name, self.smpl.graph_joints_name)

            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.h36m_eval_joint, :]
            mesh_gt_cam -= np.dot(self.joint_regressor, mesh_gt_cam)[0, None, :]

            # TEMP: use PositionNet output
            # pose_out_img = out['joint_img']
            # pose_out_img = denorm_joints(pose_out_img, out['bb2img_trans'])
            # pose_out_img[:, 2] = (pose_out_img[:, 2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2) + root_h36m_gt[None, 2]
            # pose_out_cam = pixel2cam(pose_out_img, annot['cam_param']['focal'], annot['cam_param']['princpt'])
            # pose_coord_out_h36m = transform_joint_to_other_db(pose_out_cam, self.smpl.graph_joints_name, self.h36m_joints_name)

            # h36m joint from output mesh
            mesh_out_cam = out['smpl_mesh_cam']
            pose_coord_out_h36m = np.dot(self.h36m_joint_regressor, mesh_out_cam)
            # # debug
            # pose_out_img = cam2pixel(pose_coord_out_h36m + root_h36m_gt, annot['cam_param']['focal'], annot['cam_param']['princpt'])
            # pose_out_img = transform_joint_to_other_db(pose_out_img, self.h36m_joints_name, self.smpl.graph_joints_name)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.h36m_root_joint_idx, None]  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[self.h36m_eval_joint, :]
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)

            eval_result['mpjpe'].append(np.sqrt(
                np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m) ** 2,
                                                          1)).mean() * 1000)  # meter -> milimeter
            mesh_out_cam -= np.dot(self.joint_regressor, mesh_out_cam)[0, None, :]

            # compute MPVPE
            mesh_error = np.sqrt(np.sum((mesh_gt_cam - mesh_out_cam) ** 2, 1)).mean() * 1000
            eval_result['mpvpe'].append(mesh_error)

            if cfg.render:
                img = cv2.imread(annot['img_path'])
                mesh_cam_render = out['mesh_cam_render']
                bbox = out['bbox']
                princpt = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
                img = vis_bbox(img, bbox, alpha=1)

                rendered_img = render_mesh(img, mesh_cam_render, self.face, {'focal': cfg.focal, 'princpt': princpt})

                cv2.imshow(annot['img_path'], rendered_img/255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            if cfg.vis:
                img = cv2.imread(annot['img_path'])
                bbox_to_vis = out['bbox']

                # vis input 2d pose
                # pose_out_img = out['input_joints']
                # pose_out_img = denorm_joints(pose_out_img, out['bb2img_trans'])
                # pose_scores = pose_out_img[:, 2:].round(3)
                # newimg = vis_keypoints_with_skeleton(img.copy(), pose_out_img.T, self.skeleton, kp_thresh=self.openpose_thr, alpha=1, kps_scores=pose_scores)
                # newimg = vis_bbox(newimg, bbox_to_vis, alpha=1)
                # cv2.imwrite(f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_input_2dpose.jpg', newimg)

                # vis PositionNet output
                pose_out_img = out['joint_img']
                pose_scores = (out['joint_score']).round(3)
                pose_out_img = denorm_joints(pose_out_img, out['bb2img_trans'])
                pose_out_img = np.concatenate((pose_out_img, pose_out_img[:, :1]), axis=1)
                newimg = vis_keypoints_with_skeleton(img.copy(), pose_out_img.T, self.smpl.graph_skeleton, kp_thresh=0.4, alpha=1, kps_scores=pose_scores)
                newimg = vis_bbox(newimg, bbox_to_vis, alpha=1)
                cv2.imwrite(f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_positionnet.jpg', newimg)

                # vis RotationNet output
                pose_out_img = out['joint_proj']

                pose_out_img = denorm_joints(pose_out_img, out['bb2img_trans'])
                pose_out_img = np.concatenate((pose_out_img, pose_out_img[:, :1]), axis=1)
                newimg = vis_keypoints_with_skeleton(img.copy(), pose_out_img.T, self.skeleton,
                                                     kp_thresh=0.4, alpha=1)
                newimg = vis_bbox(newimg, bbox_to_vis, alpha=1)
                cv2.imwrite(f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_final.jpg', newimg)

                save_obj(mesh_out_cam, self.face, f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_final.obj')

                # vis gt
                pose_gt_img[:, 2] = 1
                newimg = vis_keypoints_with_skeleton(img.copy(), pose_gt_img.T, self.smpl.graph_skeleton,
                                                     kp_thresh=0.4, alpha=1)
                newimg = vis_bbox(newimg, bbox_to_vis, alpha=1)
                cv2.imwrite(f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_gt.jpg', newimg)

                save_obj(mesh_gt_cam, self.face, f'./{annot["img_path"].split("_")[-1][:-4]}_{out["aid"]}_gt.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        mpjpe = np.mean(eval_result['mpjpe'])
        pa_mpjpe = np.mean(eval_result['pa_mpjpe'])
        mpvpe = np.mean(eval_result['mpvpe'])
        print('MPJPE from mesh: %.2f mm' % mpjpe)
        print('PA MPJPE from mesh: %.2f mm' % pa_mpjpe)
        print('MPVPE from mesh: %.2f mm' % mpvpe)
        return {
            'mpjpe': mpjpe,
            'pa_mpjpe': pa_mpjpe,
            'mpvpe': mpvpe,
        }



