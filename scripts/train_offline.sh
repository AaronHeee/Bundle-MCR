DATA=steam

for D in 0
do
    for H in 4 
    do
        for L in 2
        do
        CUDA_VISIBLE_DEVICES=$1 python train_offline.py \
            --dropout ${D} \
            --lr 1e-4 \
            --batch_size 32 \
            --data ${DATA} \
            --ckpt_dir "${DATA}/layer_${L}_head_${H}_dropout_${D}" \
            --n_layers ${L} \
            --n_heads ${H} \
            --seed ${2}
        done
    done
done