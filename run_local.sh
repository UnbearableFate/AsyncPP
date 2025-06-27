#!/bin/bash

# cd /work/NBB/yu_mingzhe/AsyncPP
# # module load openmpi/5.0.7/gcc11.4.0-cuda12.8.1
# conda activate py313
nnodes=8
ngpus=8
d="wikitext-103-v1"
outdir="${d}_gptn_512_768_12_8_b8"

# 参数分离
batch_args=( -b 8 --eval-batch-size 8 )
epoch_args=( --epochs 5 )
minibatch_args=( --num_minibatches 1000 --num_eval_minibatches 25 )
model_args=( --module models.gptn.gpus=${ngpus} --block_size 64 --n_embd 768 --n_head 3 --n_layer 8 --config_path models/gptn/gpus=${ngpus}/mp_conf.json )
lr_args=( --lr 3e-4 --lr_warmup --optimizer nadamw )
logtb_args=( --log_tb --tb_dir ./runs )
grad_args=( --clip_grad 10 )
dist_args=( --master_addr localhost --distributed_backend gloo )
recompute_args=( --recompute --lr_policy cosine )

# Ours
method_args=( --momentum 0.99 --optimizer nadamw )
expname="${outdir}/gpus=${ngpus}/ours/"
ckptdir="${d}/${expname}"
mkdir -p "${ckptdir}"
mpirun -np ${ngpus} python main_with_runtime.py \
    "${model_args[@]}" "${batch_args[@]}" -d "${d}" "${dist_args[@]}" \
    "${lr_args[@]}" "${epoch_args[@]}" "${minibatch_args[@]}" \
    "${grad_args[@]}" "${logtb_args[@]}" "${recompute_args[@]}" \
    "${method_args[@]}" --exp_name "${expname}" --checkpoint_dir "${ckptdir}"