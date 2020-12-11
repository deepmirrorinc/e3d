''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py

Modified by Haotian Tang
Date: August 2020
'''

import argparse
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement

import torch
from torchsparse.utils import sparse_quantize, sparse_collate
from torchsparse import SparseTensor
from model_zoo import minkunet, spvcnn, spvnas_specialized


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.05):
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)

    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels & 0xFFFF].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud

    if input_labels is not None:
        out_pc = input_point_cloud[labels_ != labels_.max(), :3]
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud
        pc_ = pc_

    inds, labels, inverse_map = sparse_quantize(pc_,
                                                feat_,
                                                labels_,
                                                return_index=True,
                                                return_invs=True)
    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int()
    )
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


# mlab.options.offscreen = True

def create_label_map(num_classes=19):
    name_label_mapping = {
        'unlabeled': 0, 'outlier': 1, 'car': 10, 'bicycle': 11,
        'bus': 13, 'motorcycle': 15, 'on-rails': 16, 'truck': 18,
        'other-vehicle': 20, 'person': 30, 'bicyclist': 31,
        'motorcyclist': 32, 'road': 40, 'parking': 44,
        'sidewalk': 48, 'other-ground': 49, 'building': 50,
        'fence': 51, 'other-structure': 52, 'lane-marking': 60,
        'vegetation': 70, 'trunk': 71, 'terrain': 72, 'pole': 80,
        'traffic-sign': 81, 'other-object': 99, 'moving-car': 252,
        'moving-bicyclist': 253, 'moving-person': 254, 'moving-motorcyclist': 255,
        'moving-on-rails': 256, 'moving-bus': 257, 'moving-truck': 258,
        'moving-other-vehicle': 259
    }

    for k in name_label_mapping:
        name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        16: 'terrain', 17: 'pole', 18: 'traffic-sign'
    }

    label_map = np.zeros(260)+num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        label_map[name_label_mapping[cls_name]] = min(num_classes, i)
    return label_map.astype(np.int64)

cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--velodyne-dir', type=str, default='sample_data')
    parser.add_argument('--pcd-path', type=str, default='sample_data')
    parser.add_argument('--write-path', type=str, default='')
    parser.add_argument('--model', type=str, default='SemanticKITTI_val_SPVNAS@35GMACs')
    args = parser.parse_args()

    # Load Model
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    print(args.model)
    print(args.pcd_path)
    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError

    model = model.to(device)

    vis_dir = os.path.join(args.write_path, "vis")
    label_dir = os.path.join(args.write_path, "label")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    pcd_files = os.listdir(args.pcd_path)
    for fi in tqdm(pcd_files, total=len(pcd_files)):
        pc = np.array([])
        pcd = o3d.io.read_point_cloud(os.path.join(args.pcd_path, fi))

        input_pcd = o3d.geometry.PointCloud()
        pts = [p for p in pcd.points if all(-10000 < v < 10000 for v in p)]
        input_pcd.points = o3d.utility.Vector3dVector(pts)
        pc = np.array([p for p in input_pcd.points])
        pc = np.hstack([pc, [[0]] * len(pc)])
        pc.astype(np.float32)

        feed_dict = process_point_cloud(pc)
        inputs = feed_dict['lidar'].to(device)
        outputs = model(inputs)
        predictions = outputs.argmax(1).cpu().numpy()
        predictions = predictions[feed_dict['inverse_map']]

        # Visualization ply 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(feed_dict['pc'][:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([cmap[p][:3] for p in predictions]) / 256.0)
        o3d.io.write_point_cloud(os.path.join(vis_dir, fi), pcd)

        # Label ply
        label_pcd = np.array([(p[0], p[1], p[2], predictions[i]) for i, p in enumerate(feed_dict['pc'][:, :3].tolist())],
                             dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('label', 'u1')])
        el = PlyElement.describe(label_pcd, 'points_with_semantic_label')
        PlyData([el]).write(os.path.join(label_dir, fi))
