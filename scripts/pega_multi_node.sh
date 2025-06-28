#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 8
#PBS -l elapstim_req=02:59:00
#PBS -v NQSV_MPI_VER=5.0.7/gcc11.4.0-cuda12.8.1
#PBS -M kanakawapanman@gmail.com

module load openmpi/5.0.7/gcc11.4.0-cuda12.8.1

export CUDA_VISIBLE_DEVICES=0

# compute world size
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# master address and port
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=6003

# compute local node-rank
hosts=( $(sort -u "$PBS_NODEFILE") )
NODE_RANK=0
for idx in "${!hosts[@]}"; do
  if [[ "${hosts[idx]}" == "$(hostname)" ]]; then
    NODE_RANK=$idx
    break
  fi
done

echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"

timestamp=$(date "+%Y%m%d%H%M%S")

ngpus=8
d="wikitext-103-v1"
outdir="${d}_gptn_512_768_12_8_b8"

work_dir=/work/NBB/yu_mingzhe/AsyncPP

# 参数分离
batch_args=( -b 8 --eval-batch-size 8 )
epoch_args=( --epochs 50 )
minibatch_args=( --num_minibatches 1000 --num_eval_minibatches 25 )
model_args=( --module models.gptn.gpus=${ngpus} --block_size 512 --n_embd 768 --n_head 12 --n_layer 8 --config_path ${work_dir}/models/gptn/gpus=${ngpus}/mp_conf.json )
lr_args=( --lr 3e-4 --lr_warmup --optimizer nadamw )
logtb_args=( --log_tb --tb_dir ${work_dir}/runs )
grad_args=( --clip_grad 10 )
dist_args=( --distributed_backend nccl )
recompute_args=( --recompute --lr_policy cosine )

# Ours
method_args=( --momentum 0.99 --optimizer nadamw )
expname="${outdir}/gpus=${ngpus}/ours/"
ckptdir="${work_dir}/check_point${d}/${expname}"

if [[ "$NODE_RANK" -eq 0 ]]; then
  mkdir -p "${ckptdir}"
  echo "Exp name: ${expname}"
  echo "Checkpoint dir: ${ckptdir}"
fi

export MASTER_ADDR
export MASTER_PORT
export CUDA_VISIBLE_DEVICES

mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -x MASTER_ADDR -x MASTER_PORT -x CUDA_VISIBLE_DEVICES \
 -np $WORLD_SIZE --map-by ppr:$NPROC_PER_NODE:node --report-bindings \
  /work/NBB/yu_mingzhe/miniconda3/envs/py313/bin/python ${work_dir}/main_with_runtime.py \
    "${model_args[@]}" "${batch_args[@]}" -d "${d}" "${dist_args[@]}" \
    "${lr_args[@]}" "${epoch_args[@]}" "${minibatch_args[@]}" \
    "${grad_args[@]}" "${logtb_args[@]}" "${recompute_args[@]}" \
    "${method_args[@]}" --exp_name "${expname}" --checkpoint_dir "${ckptdir}"