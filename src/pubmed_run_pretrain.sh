PROJ_DIR=..

DOMAIN=pubmed
MAX_LENGTH=64
BATCH_SIZE=32
PROCESSED_DIR=$PROJ_DIR/data/$DOMAIN
LOG_DIR=$PROJ_DIR/logs/$DOMAIN
CHECKPOINT_DIR=$PROJ_DIR/ckpt/$DOMAIN

LR="1e-6"
# MODEL_TYPE=graphformer
MODEL_TYPE=contextualgraphformer

echo "start training..."

python -m torch.distributed.launch --nproc_per_node=4 --master_port 19298 \
    -m OpenLP.driver.patton_pretrain  \
    --output_dir $CHECKPOINT_DIR/patton/$MODEL_TYPE/$LR  \
    --model_name_or_path "bert-base-uncased"  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 200  \
    --eval_steps 100  \
    --logging_steps 10 \
    --train_path $PROCESSED_DIR/train.text.jsonl  \
    --eval_path $PROCESSED_DIR/val.text.jsonl  \
    --fp16  \
    --per_device_train_batch_size $BATCH_SIZE  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len $MAX_LENGTH  \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/patton/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --mlm_loss True \
    --dataloader_num_workers 32
