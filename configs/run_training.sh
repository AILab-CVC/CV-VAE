pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
project_root=./
cd $project_root && pip install -e .

HOST_NUM=1 # 总的服务器数量
INDEX=0 # 当前服务器的编号
CHIEF_IP=127.0.0.1 # 主服务器的ip
HOST_GPU_NUM=8


config_file=configs/cvvae_sd3_constraint_training.yaml
save_root=SAVE_ROOT
expr_name=EXPR_NAME

mkdir -p $save_root/$expr_name
mkdir -p $save_root/$expr_name/stdout

torchrun --nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=$CHIEF_IP --master_port=20002 --node_rank=$INDEX main.py \
    --base $config_file \
    --train \
    --name $expr_name \
    --logdir $save_root \
    --devices $HOST_GPU_NUM \
    --seed 129 \
    --wandb true \
    lightning.trainer.num_nodes=$HOST_NUM \