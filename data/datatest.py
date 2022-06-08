import numpy as np
from nuscenes.nuscenes import NuScenes
# print('"')
# dataset = NuScenes(version='v1.0-trainval',dataroot='/sharedata/home/shared/jiangq/nuscenes')
# mapping = dataset.lidarseg_name2idx_mapping
# for name, color in dataset.colormap.items():
#     print(str(mapping[name])+':', list(color))

for i in range(16):
    print(str(i)+': '+str(i))