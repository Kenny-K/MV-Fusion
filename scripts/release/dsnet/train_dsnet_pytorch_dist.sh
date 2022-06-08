ngpu=2
tag=slot_0200
# export CUDA_VISIBLE_DEVICES=1
python trial_train.py --batch_size 1 --config cfgs/release/fusion.yaml\
             --tag ${tag}\
             --pretrained_ckpt pretrained_weight/kitti_v121.pth\
             --nofix


#  0303 - slot!
#  0403 - slot
#  0301 - slot
#  0300 - nuscenes baseline ?



# python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
#     --tcp_port 12345 \
#     --batch_size ${ngpu} \
#     --config cfgs/release/dsnet.yaml \
#     --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
#     --tag ${tag} \
#     --launcher pytorch \
#     --fix_semantic_instance

# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nproc_per_node=${ngpu} trial_train.py \
#     --tcp_port 2345 \
#     --batch_size ${ngpu} \
#     --config cfgs/release/fusion.yaml \
#     --tag ${tag} \
    # --launcher pytorch \
# export LD_LIBRARY_PATH='usr/local/cuda/lib64:/usr/local/cuda-11.4/lib64'
