SEED=171
mode=inter
N=5
K=1

python3 main.py \
    --seed=${SEED} \
    --mode=${mode} \
    --N=${N} \
    --K=${K} \
    --similar_k=10 \
    --eval_every_meta_steps=100 \
    --name=TEST_TYPE \
    --train_mode=type \
    --eval_mode=add \
    --inner_steps=2 \
    --inner_size=32 \
    --max_ft_steps=3 \
    --lambda_max_loss=2 \
    --inner_lambda_max_loss=5 \
    --tagging_scheme=BIOES \
    --viterbi=hard \
    --concat_types=None \
    --ignore_eval_test \
    --data_path data/episode-data \
    --max_meta_steps=1000 \
    --eval_every_meta_steps=40 \