#train
#python train.py -n 1 -g 8 -nr 0 -lr 1e-1

# Horovod
#horovodrun -np 1 -H localhost:1 python train_h.py

# dist train
# export CUDA_VISIBLE_DEVICES=3
# python train.py -n 2 -nr 0 2>&1 & > train.log
# export CUDA_VISIBLE_DEVICES=2
# python train.py -n 2 -nr 1 2>&1 & > rank1.log

python gan/train_dcgan.py