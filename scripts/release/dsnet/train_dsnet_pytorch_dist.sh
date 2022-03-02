ngpu=4
tag=train_dsnet_pytorch_dist

# python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
#     --tcp_port 12345 \
#     --batch_size ${ngpu} \
#     --config cfgs/release/dsnet.yaml \
#     --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
#     --tag ${tag} \
#     --launcher pytorch \
#     --fix_semantic_instance


python -m torch.distributed.run --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port 12345 \
    --batch_size ${ngpu} \
    --config cfgs/release/dsnet.yaml \
    --tag ${tag} \
    --launcher pytorch \
    --fix_semantic_instance

# python -m torch.distributed.launch --nproc_per_node=2 cfg_train.py --tcp_port 12345 --batch_size 2 --config cfgs/release/dsnet.yaml --tag train_dsnet_pytorch_dist --launcher pytorch