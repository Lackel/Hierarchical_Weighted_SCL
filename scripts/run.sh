#!/usr/bin bash
for s in 0 1 2 3 4
do 
    python wscl.py \
        --dataset hwu64 \
        --seed $s \
        --freeze_bert_parameters
done
