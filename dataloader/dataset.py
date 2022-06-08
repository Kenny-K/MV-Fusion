#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
import pickle
from torch.utils import data
from tqdm import tqdm
from scipy import stats as s

from nuscenes.nuscenes import NuScenes as NuSc
from nuscenes.utils.data_io import load_bin_file

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

# things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person', 'bicyclist', 'motorcyclist']
# stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
# things_ids = []
# for i in sorted(list(semkittiyaml['labels'].keys())):
#     if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in things:
#         things_ids.append(i)

# print(things_ids)

class NuScenes(data.Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, return_ins = False):
        self.return_ref = return_ref
        self.return_ins = return_ins
        self.data_path = data_path
        self.imageset = imageset
        self.learning_map = {
            1: 0,
            5: 0,
            7: 0,
            8: 0,
            10:0,
            11:0,
            13:0,
            19:0,
            20:0,
            0: 0,
            29:0,
            31:0,
            9: 1,
            14:2,
            15:3,
            16:3,
            17:4,
            18:5,
            21:6,
            2: 7,
            3: 7,
            4: 7,
            6: 7,
            12:8,
            22:9,
            23:10,
            24:11,
            25:12,
            26:13,
            27:14,
            28:15,
            30:16
        }

        if imageset == 'test':
            self.dataset = NuSc(version='v1.0-test', dataroot=self.data_path, verbose=True)
        else:
            self.dataset = NuSc(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
            self.dataset.lidarseg_idx2name_mapping

        self.token_list = []
        for sample in self.dataset.sample:
            self.token_list.append(sample['token'])
        
        if imageset == 'train':
            self.token_list = self.token_list[:28130]
        elif imageset == 'val':
            self.token_list = self.token_list[28130:]
        else:
            print('FATAL! Imageset must be train / val / test')

        self.things_ids = [9, 14, 15, 16, 17, 18, 21, 2, 3, 4, 6, 12, 22, 23]

    def __len__(self):
        return len(self.token_list)

    def read_point_cloud(self, file_name):
        '''
        pcl_file_name == 'xxxx_LIDAR_TOP_token.pcd.bin'
        return points of (x,y,z,intensity) -> pcl
        where pcl.points is a (nbr_pts,dim) np.array
        To get the ith points(x,y,z,r) pcl[i-1,:]
        '''
        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        return points

    def read_panoptic_label(self,npz_file_name):
        '''
        npz_file_name = 'token_panoptic.npz'
        return label np.array of shape (nbr_points, 1)
        '''
        bin_content = load_bin_file(npz_file_name,type='panoptic')
        return bin_content.T

    def __getitem__(self, idx):
        """
        NuScenes Panoptic Labels format: (general class index * 1000 + instance index)
        Note here general class index (32 classes in total) rather than the challenge class index (16 classes in total) is used.

        Returns:
            raw_data: (N, 4), x y z r
            batch_cnt: int
            label: int, class label.
        """

        # Read Sample, which is a key frame in a scene.
        sample = self.dataset.get('sample', self.token_list[idx])
        # Read Point Cloud
        pcl_token = sample['data']['LIDAR_TOP']
        data_file_name = self.data_path + '/' + self.dataset.get('sample_data',pcl_token)['filename']
        raw_data = self.read_point_cloud(data_file_name)  #(N,4)

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            sem_labels = annotated_data
            ins_labels = annotated_data
            valid = annotated_data
        else:
            label_file_name = self.data_path + '/panoptic/v1.0-trainval/' + pcl_token + '_panoptic.npz'
            annotated_data = self.read_panoptic_label(label_file_name)
            annotated_data = annotated_data.reshape(-1,1)
            assert annotated_data.shape[0] == raw_data.shape[0], print("WARNING! Labels not matched with points!")
            sem_labels = (annotated_data // 1000).astype(np.int64)
            ins_labels = (annotated_data % 1000).astype(np.int64)
            valid = np.isin(sem_labels, self.things_ids).reshape(-1)
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)
        data_tuple = (raw_data[:,:3], sem_labels.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        if self.return_ins:
            data_tuple += (ins_labels, valid)
        data_tuple += (self.token_list[idx],)
        return data_tuple   # ( x y z sem_label ref ins_label valid im_index ) length: 8


class SemKITTI(data.Dataset):
    def __init__(self, data_path, imageset = 'train', return_ref = False, return_ins = False):
        self.return_ref = return_ref
        self.return_ins = return_ins
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path,str(i_folder).zfill(2),'velodyne']))
        self.im_idx.sort()

        self.things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person', 'bicyclist', 'motorcyclist']
        self.stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
        self.things_ids = []
        for i in sorted(list(semkittiyaml['labels'].keys())):
            if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in self.things:
                self.things_ids.append(i)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        # print("loading {}, shape {}".format(self.im_idx[index], raw_data.shape))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:,0],dtype=int),axis=1)
            sem_labels = annotated_data
            ins_labels = annotated_data
            valid = annotated_data
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
            # ins_labels = (annotated_data & 0xFFFF0000) >> 16 # different classes could use same ins ids
            ins_labels = annotated_data
            # valid = (((ins_labels & 0xFFFF0000) >> 16) != 0).reshape(-1) # TODO: maybe this is not ok
            valid = np.isin(sem_labels, self.things_ids).reshape(-1) # use 0 to filter out valid indexes is enough
            # print(np.sum(valid) - np.sum((((ins_labels & 0xFFFF0000) >> 16) != 0)))
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)
        data_tuple = (raw_data[:,:3], sem_labels.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:,3],)
        if self.return_ins:
            data_tuple += (ins_labels, valid)
        data_tuple += (self.im_idx[index],)
        return data_tuple   # ( x y z sem_label ref ins_label valid im_index ) length: 8

    def count_ins(self):
        pbar = tqdm(total=len(self.im_idx), dynamic_ncols=True)
        counter = np.zeros([9], dtype=np.int32)
        min_valid_pn = 10000086
        max_valid_pn = -1
        for i in range(len(self.im_idx)):
            # raw_data = np.fromfile(self.im_idx[i], dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(self.im_idx[i].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            _sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
            ins_labels = annotated_data
            sem_labels = np.vectorize(self.learning_map.__getitem__)(_sem_labels)
            for j in range(1,9):
                j_ind = (sem_labels == j)
                j_ins_labels = ins_labels[j_ind]
                counter[j] += np.unique(j_ins_labels).reshape(-1).shape[0]
            pbar.update(1)
            valid_pn = np.sum(np.isin(_sem_labels, self.things_ids).reshape(-1))
            if valid_pn > max_valid_pn:
                max_valid_pn = valid_pn
            if valid_pn < min_valid_pn:
                min_valid_pn = valid_pn
            print(valid_pn, sem_labels.shape[0])
        pbar.close()
        counter = counter[1:]
        print("Counting results: ")
        print(counter)
        counter = counter.astype(np.float32)
        counter /= (np.min(counter) if np.min(counter) != 0 else 1.0)
        print("Weights: ")
        print(counter)
        print("max_valid_pn: {}".format(max_valid_pn))
        print("min_valid_pn: {}".format(min_valid_pn))

    def count_box_size(self):
        pbar = tqdm(total=len(self.im_idx), dynamic_ncols=True)
        counter = np.zeros([9], dtype=np.float32)
        mean_size = np.zeros([9, 2], dtype=np.float32)
        max_size = np.zeros([9, 2], dtype=np.float32)
        min_size = np.zeros([9, 2], dtype=np.float32) + 10086
        for i in range(len(self.im_idx)):
            #if i % 10 != 0:
            #    pbar.update(1)
            #    continue
            raw_data = np.fromfile(self.im_idx[i], dtype=np.float32).reshape((-1, 4))
            annotated_data = np.fromfile(self.im_idx[i].replace('velodyne','labels')[:-3]+'label', dtype=np.int32).reshape((-1,1))
            _sem_labels = annotated_data & 0xFFFF #delete high 16 digits binary
            ins_labels = annotated_data
            sem_labels = np.vectorize(self.learning_map.__getitem__)(_sem_labels)
            pbar.update(1)
            for j in range(1, 9):
                j_ind = (sem_labels == j)
                j_ins_labels = ins_labels[j_ind]
                for j_ins_lab in np.unique(j_ins_labels):
                    j_pcd = raw_data[(ins_labels == j_ins_lab).reshape(-1)]
                    if j_pcd.shape[0] < 50:
                        continue
                    x = j_pcd[:, 0].max() - j_pcd[:, 0].min()
                    y = j_pcd[:, 1].max() - j_pcd[:, 1].min()
                    if x < y:
                        tmp = x
                        x = y
                        y = tmp
                    mean_size[j, 0] += x
                    mean_size[j, 1] += y
                    counter[j] += 1
                    if x > max_size[j, 0]:
                        max_size[j, 0] = x
                    if y > max_size[j, 1]:
                        max_size[j, 1] = y
                    if x < min_size[j, 0]:
                        min_size[j, 0] = x
                    if y < min_size[j, 1]:
                        min_size[j, 1] = y
        pbar.close()
        counter[0] = 1
        print("Mean Size: {}".format(mean_size / counter.reshape(-1, 1)))
        print("Max Size: {}".format(max_size))
        print("Min Size: {}".format(min_size))

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

class voxel_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False, ignore_label = 255, return_test = False,
            fixed_volume_space= False, max_volume_space = [50,50,1.5], min_volume_space = [-50,-50,-3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 4:
            raise Exception('Not implement instance label for voxel_dataset')
        else: raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        max_bound = np.percentile(xyz,100,axis = 0)
        min_bound = np.percentile(xyz,0,axis = 0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range/(cur_grid_size-1)
        if (intervals==0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz,min_bound,max_bound)-min_bound)/intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)

        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz,xyz),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea)
        return data_tuple

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0]*np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0]*np.sin(input_xyz_polar[1])
    return np.stack((x,y,input_xyz_polar[2]),axis=0)

class spherical_dataset(data.Dataset):
  def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False,
               scale_aug =False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
               min_rad=-np.pi/4, max_rad=np.pi/4, ignore_label = 255,
               return_test = False, fixed_volume_space= False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               center_type='Axis_center', H=None, W=None, fov_up=None, fov_down=None):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.noise_rotation = np.random.uniform(min_rad, max_rad)

        assert center_type in ['Axis_center', 'Mass_center']
        self.center_type = center_type

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

  def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz,labels,sig,ins_labels,valid,pcd_fname = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 7:
            xyz,labels,sig,ins_labels,valid,pcd_fname,minicluster = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:,0] = noise_scale * xyz[:,0]
            xyz[:,1] = noise_scale * xyz[:,1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                np.random.normal(0, self.trans_std[1], 1),
                                np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers #TODO: calculate relative coordinate using polar system?
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea) # (grid-wise coor, grid-wise sem label, point-wise grid index, point-wise sem label, [relative polar coor(3), polar coor(3), cat coor(2), ref signal(1)])

        if len(data) == 6:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname) # plus (point-wise instance label, point-wise center offset)

        if len(data) == 7:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, minicluster) # plus (point-wise instance label, point-wise center offset)

        return data_tuple


class fusion_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False,
               scale_aug =False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
               min_rad=-np.pi/4, max_rad=np.pi/4, ignore_label = 255,
               return_test = False, fixed_volume_space= False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               center_type='Axis_center', H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.noise_rotation = np.random.uniform(min_rad, max_rad)

        assert center_type in ['Axis_center', 'Mass_center']
        self.center_type = center_type

        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                    dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                dtype=np.int32)       # [H,W] mask

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz,labels,sig,ins_labels,valid,pcd_fname = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 7:
            xyz,labels,sig,ins_labels,valid,pcd_fname,minicluster = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:,0] = noise_scale * xyz[:,0]
            xyz[:,1] = noise_scale * xyz[:,1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                np.random.normal(0, self.trans_std[1], 1),
                                np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate

        # get range view
        self.points = xyz
        self.remissions = sig
        self.do_range_projection()
        range_image = np.concatenate([self.proj_range[:,:, np.newaxis], self.proj_xyz, self.proj_remission[:,:,np.newaxis]], axis=-1)
        range_image = np.transpose(range_image, (2,0,1))        # [C, H, W]
        pt_pix_mapping = np.concatenate([self.proj_y[:, np.newaxis], self.proj_x[:, np.newaxis]], axis=1)

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers #TODO: calculate relative coordinate using polar system?
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea) # (grid-wise coor, grid-wise sem label, point-wise grid index, point-wise sem label, [relative polar coor(3), polar coor(3), cat coor(2), ref signal(1)])

        if len(data) == 6:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, range_image, pt_pix_mapping, pcd_fname) # plus (point-wise instance label, point-wise center offset)

        if len(data) == 7:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, minicluster) # plus (point-wise instance label, point-wise center offset)

        return data_tuple


class fusion_half_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug = False, flip_aug = False,
               scale_aug =False, transform_aug=False, trans_std=[0.1, 0.1, 0.1],
               min_rad=-np.pi/4, max_rad=np.pi/4, ignore_label = 255,
               return_test = False, fixed_volume_space= False,
               max_volume_space = [50,np.pi,1.5], min_volume_space = [3,-np.pi,-3],
               center_type='Axis_center', H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.half_grid_size = np.asarray(grid_size)
        self.grid_size = np.copy(self.half_grid_size)
        self.grid_size[1] = self.grid_size[1] * 2
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

        self.scale_aug = scale_aug
        self.transform = transform_aug
        self.trans_std = trans_std
        self.noise_rotation = np.random.uniform(min_rad, max_rad)

        assert center_type in ['Axis_center', 'Mass_center']
        self.center_type = center_type

        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                    dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                dtype=np.int32)       # [H,W] mask

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz,labels = data
        elif len(data) == 3:
            xyz,labels,sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 6:
            xyz,labels,sig,ins_labels,valid,pcd_fname = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        elif len(data) == 7:
            xyz,labels,sig,ins_labels,valid,pcd_fname,minicluster = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else: raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random()*360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot( xyz[:,:2],j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4,1)
            if flip_type==1:
                xyz[:,0] = -xyz[:,0]
            elif flip_type==2:
                xyz[:,1] = -xyz[:,1]
            elif flip_type==3:
                xyz[:,:2] = -xyz[:,:2]

        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:,0] = noise_scale * xyz[:,0]
            xyz[:,1] = noise_scale * xyz[:,1]

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                np.random.normal(0, self.trans_std[1], 1),
                                np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate

        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:,0],100,axis = 0)
        min_bound_r = np.percentile(xyz_pol[:,0],0,axis = 0)
        max_bound = np.max(xyz_pol[:,1:],axis = 0)
        min_bound = np.min(xyz_pol[:,1:],axis = 0)
        max_bound = np.concatenate(([max_bound_r],max_bound))
        min_bound = np.concatenate(([min_bound_r],min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range/(cur_grid_size-1) # (size-1) could directly get index starting from 0, very convenient

        if (intervals==0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol,min_bound,max_bound)-min_bound)/intervals)).astype(np.int) # point-wise grid index

        half_mask = grid_ind[:,1] < 180
        # if np.random.rand() > 0.5:
        #     half_mask = grid_ind[:,1] < 180
        # else:
        #     half_mask = grid_ind[:,1] >= 180
        grid_ind = grid_ind[half_mask]
        labels = labels[half_mask]
        xyz_pol = xyz_pol[half_mask]
        xyz = xyz[half_mask]
        sig = sig[half_mask]
        valid = valid[half_mask]
        ins_labels = ins_labels[half_mask]
        # print(np.amax(grid_ind,axis=0))
        # print(np.amin(grid_ind,axis=0))

        # get range view
        self.points = xyz
        self.remissions = sig
        self.do_range_projection()
        range_image = np.concatenate([self.proj_range[:,:, np.newaxis], self.proj_xyz, self.proj_remission[:,:,np.newaxis]], axis=-1)
        range_image = np.transpose(range_image, (2,0,1))        # [C, H, W]
        pt_pix_mapping = np.concatenate([self.proj_y[:, np.newaxis], self.proj_x[:, np.newaxis]], axis=1)

        # process voxel position
        voxel_position = np.zeros(self.grid_size,dtype = np.float32)
        dim_array = np.ones(len(self.grid_size)+1,int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size)*intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        # process labels
        processed_label = np.ones(self.half_grid_size,dtype = np.uint8)*self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind,labels],axis = 1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:,0],grid_ind[:,1],grid_ind[:,2])),:]
        processed_label = nb_process_label(np.copy(processed_label),label_voxel_pair)
        data_tuple = (voxel_position,processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + min_bound
        return_xyz = xyz_pol - voxel_centers #TODO: calculate relative coordinate using polar system?
        return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]),axis = 1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) >= 3:
            return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]),axis = 1)

        if self.return_test:
            data_tuple += (grid_ind,labels,return_fea,index)
        else:
            data_tuple += (grid_ind,labels,return_fea) # (grid-wise coor, grid-wise sem label, point-wise grid index, point-wise sem label, [relative polar coor(3), polar coor(3), cat coor(2), ref signal(1)])

        if len(data) == 6:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, range_image, pt_pix_mapping, pcd_fname) # plus (point-wise instance label, point-wise center offset)

        if len(data) == 7:
            offsets = np.zeros([xyz.shape[0], 3], dtype=np.float32)
            offsets = nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, self.center_type)
            data_tuple += (ins_labels, offsets, valid, xyz, pcd_fname, minicluster) # plus (point-wise instance label, point-wise center offset)

        return data_tuple


def calc_xyz_middle(xyz):
    return np.array([
        (np.max(xyz[:, 0]) + np.min(xyz[:, 0])) / 2.0,
        (np.max(xyz[:, 1]) + np.min(xyz[:, 1])) / 2.0,
        (np.max(xyz[:, 2]) + np.min(xyz[:, 2])) / 2.0
    ], dtype=np.float32)

things_ids = set([10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])

# @nb.jit #TODO: why jit would lead to offsets all zero?
def nb_aggregate_pointwise_center_offset(offsets, xyz, ins_labels, center_type):
    # ins_num = np.max(ins_labels) + 1
    # for i in range(1, ins_num):
    for i in np.unique(ins_labels):
        # if ((i & 0xFFFF0000) >> 16) == 0: #TODO: change to use thing list to filter
        #     continue
        if (i & 0xFFFF) not in things_ids:
            continue
        i_indices = (ins_labels == i).reshape(-1)
        xyz_i = xyz[i_indices]
        if xyz_i.shape[0] <= 0:
            continue
        if center_type == 'Axis_center':
            mean_xyz = calc_xyz_middle(xyz_i)
        elif center_type == 'Mass_center':
            mean_xyz = np.mean(xyz_i, axis=0)
        else:
            raise NotImplementedError
        offsets[i_indices] = mean_xyz - xyz_i
    return offsets

@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])',nopython=True,cache=True,parallel = False)
def nb_process_label(processed_label,sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,),dtype = np.uint16)
    counter[sorted_label_voxel_pair[0,3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0,:3]
    for i in range(1,sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i,:3]
        if not np.all(np.equal(cur_ind,cur_sear_ind)):
            processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,),dtype = np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i,3]] += 1
    processed_label[cur_sear_ind[0],cur_sear_ind[1],cur_sear_ind[2]] = np.argmax(counter)
    return processed_label

def collate_fn_BEV(data): # stack alone batch dimension
    data2stack=np.stack([d[0] for d in data]).astype(np.float32) # grid-wise coor
    label2stack=np.stack([d[1] for d in data])                   # grid-wise sem label
    grid_ind_stack = [d[2] for d in data]                        # point-wise grid index
    point_label = [d[3] for d in data]                           # point-wise sem label
    xyz = [d[4] for d in data]                                   # point-wise coor

    pt_ins_labels = [d[5] for d in data]                         # point-wise instance label
    pt_offsets = [d[6] for d in data]                            # point-wise center offset
    pt_valid = [d[7] for d in data]                              # point-wise indicator for foreground points
    pt_cart_xyz = [d[8] for d in data]                           # point-wise cart coor

    return {
        'vox_coor': torch.from_numpy(data2stack),
        'vox_label': torch.from_numpy(label2stack),
        'grid': grid_ind_stack,
        'pt_labs': point_label,
        'pt_fea': xyz,
        'pt_ins_labels': pt_ins_labels,
        'pt_offsets': pt_offsets,
        'pt_valid': pt_valid,
        'pt_cart_xyz': pt_cart_xyz,
        'pcd_fname': [d[9] for d in data]
    }

def collate_fn_BEV_fusion(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32) # grid-wise coor
    label2stack=np.stack([d[1] for d in data])                   # grid-wise sem label
    grid_ind_stack = [d[2] for d in data]                        # point-wise grid index
    point_label = [d[3] for d in data]                           # point-wise sem label
    xyz = [d[4] for d in data]                                   # point-wise coor

    pt_ins_labels = [d[5] for d in data]                         # point-wise instance label
    pt_offsets = [d[6] for d in data]                            # point-wise center offset
    pt_valid = [d[7] for d in data]                              # point-wise indicator for foreground points
    pt_cart_xyz = [d[8] for d in data]                           # point-wise cart coor
    
    range_image = np.stack([d[9] for d in data], axis=0)
    pt_pix_mapping = [d[10] for d in data]

    return {
        'vox_coor': torch.from_numpy(data2stack),
        'vox_label': torch.from_numpy(label2stack),
        'grid': grid_ind_stack,
        'pt_labs': point_label,
        'pt_fea': xyz,
        'pt_ins_labels': pt_ins_labels,
        'pt_offsets': pt_offsets,
        'pt_valid': pt_valid,
        'pt_cart_xyz': pt_cart_xyz,
        'range_image': range_image,
        'pt_pix_mapping': pt_pix_mapping,
        'pcd_fname': [d[11] for d in data]
    }

def collate_fn_BEV_test(data):
    data2stack=np.stack([d[0] for d in data]).astype(np.float32)
    label2stack=np.stack([d[1] for d in data])
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack),torch.from_numpy(label2stack),grid_ind_stack,point_label,xyz,index


if __name__ == '__main__':
    dataset = SemKITTI('./sequences', 'train')
    dataset.count_box_size()
