for seed in 1 2 3 4 5
do
    python3 train.py \
      --checkpoint ../../output/checkpoints/last-v31.ckpt \
      --dataset universal-dependencies \
      --batch_size 16 \
      --epochs 10 \
      --learning_rate 3e-5 \
      --scheduler_type linear \
      --warmup_ratio 0.05 \
      --weight_decay 0.05 \
      --load_best_model \
      --seed $seed
done