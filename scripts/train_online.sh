DATA='steam'

for e in 50000; do
    for r in f1; do
        CUDA_VISIBLE_DEVICES=$1 python train_online.py \
            --pretrained_weights $3 \
            --hidden_size 32 \
            --embed_size 32 \
            --n_heads 4 \
            --n_layers 2 \
            --hidden 32 \
            --data ${DATA} \
            --verbose 1000 \
            --epochs ${e} \
            --reward ${r} \
            --ckpt_dir "${DATA}/online/bunt-${r}-epoch-${e}" \
            --command 2 \
            --max_run 10 \
            --model maskedbunt \
            --policy_strategy pretrain \
            --ask_k 1 \
            --seed ${2}
    done
done
