import numpy as np
import cv2
import random
from config import cfg
import math
from PIL import Image
import io
from einops import repeat

def load_img_from_lmdb(path: str, lmdb, order='RGB'):
    key = '/'.join(path.split('/')[-2:])
    img = lmdb.get(key.encode())
    img = Image.open(io.BytesIO(img))
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = repeat(img, 'h w -> h w c', c=3)
    if order=='RGB':
        return img
    else:
        return img[:,:,::-1].copy()

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1]
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img)

    x_center = (xmin+xmax)/2.; width = xmax-xmin
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def compute_iou(src_roi, dst_roi):
    # IoU calculate with GTs
    xmin = np.maximum(dst_roi[:, 0], src_roi[:, 0])
    ymin = np.maximum(dst_roi[:, 1], src_roi[:, 1])
    xmax = np.minimum(dst_roi[:, 0] + dst_roi[:, 2], src_roi[:, 0] + src_roi[:, 2])
    ymax = np.minimum(dst_roi[:, 1] + dst_roi[:, 3], src_roi[:, 1] + src_roi[:, 3])

    interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
    boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
    sumArea = boxAArea + boxBArea

    iou = interArea / (sumArea - interArea + 1e-5)

    return iou

# def trunc_bbox(bbox):
#     if False and random.random() >= 0.3:
#         return bbox
#     else:
#         x, y, w, h = bbox
#         x_aug_range, y_aug_range = w/2, h/2
#         x_aug, y_aug = random.random() * x_aug_range, random.random() * y_aug_range
#
#         if random.random() <= 0.5:
#             x, y = x+x_aug, y+y_aug
#         else: # good
#             w, h = w-x_aug, h-y_aug
#
#     return [x,y,w,h]

def process_bbox(bbox, img_width, img_height, is_3dpw_test=False):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if is_3dpw_test:
        bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
    else:
        if w*h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
        else:
            return None

    # aspect ratio preserving bbox
    if is_3dpw_test:
        scale = 1.1
    else:
        scale = np.random.rand() / 5 + 1 # 1.0 ~ 1.2
    # print(scale)
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*scale
    bbox[3] = h*scale
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def get_aug_config(exclude_flip):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip


def augmentation(img, bbox, data_split, exclude_flip=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip,)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)

    # cv2.imshow('input', img/255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

