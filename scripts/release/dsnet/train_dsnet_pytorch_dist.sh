ngpu=2
tag=train_noDS_pytorch_dist

# python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
#     --tcp_port 12345 \
#     --batch_size ${ngpu} \
#     --config cfgs/release/dsnet.yaml \
#     --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
#     --tag ${tag} \
#     --launcher pytorch \
#     --fix_semantic_instance


# torchrun --nproc_per_node=${ngpu} trial_train.py \
#     --tcp_port 2345 \
#     --batch_size ${ngpu} \
#     --config cfgs/release/dsnet.yaml \
#     --tag ${tag} \
#     --launcher pytorch \

python trial_train.py --batch_size 1 --config cfgs/release/dsnet.yaml --tag train_dsnet_pytorch