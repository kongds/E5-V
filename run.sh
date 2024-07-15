RUN=e5v-8b

args=()

BASE_MODEL="models/llava-llama-3-8b"
TEMPLATE='*sent_0*\nSummary_above_sentence_in_one_word:'

BIT=4

R=64
ALPHA=16
BATCH_SIZE=768
MICRO_BATCH_SIZE=24
EPOCH=2
LR=4e-4

echo $BASE_MODEL
echo $TEMPLATE


echo $MICRO_BATCH_SIZE $BATCH_SIZE

GPUS=8
NUM_NODES=4

wandb online


NCCL_DEBUG=ERROR deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES ft_llm.py \
        --base_model   $BASE_MODEL \
        --data_path 'data/nli_for_simcse.csv' \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --learning_rate $LR \
        --cutoff_len 32 \
        --lora_r $R \
        --lora_alpha $ALPHA \
        --lora_dropout 0.05 \
        --output_dir $RUN  --is_sentemb \
        --mask_embedding_sentence_template $TEMPLATE --use_neg_sentence --save_steps 50 \
        --deepspeed ds.config \
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj  --logging_steps 1 --grad_checkpoint \
         --load_kbit $BIT \
         ${args[@]}

