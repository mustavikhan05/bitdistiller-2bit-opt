export MODEL_PATH='/content/drive/MyDrive/Bitdistiller-OPT-Quant/models/opt-125m'
export TEACHER_MODEL_PATH='facebook/opt-125m' 
export SAVE_PATH=$2
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --num_gpus=1 train.py \
    --model_name_or_path $MODEL_PATH \
    --teacher_model_name_or_path $TEACHER_MODEL_PATH \
    --data_path $1 \
    --model_max_length 2048 \
    --output_dir $SAVE_PATH \
    --logging_dir $3 \
    --num_train_epochs $4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 4 \
    --save_strategy "no" \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 128 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip /content/drive/MyDrive/Bitdistiller-OPT-Quant/quantization/clip_cache/opt-125m/int2-g128.pt
