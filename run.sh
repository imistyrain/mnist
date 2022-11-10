if [ $# -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi
# train
#python train.py
# ddp trian
#python train_ddp.py

# Horovod 1
#horovodrun -np 1 -H localhost:1 python train_hvd.py
# Horovod 2
horovodrun -np 2 -H localhost:2 python train_hvd.py
# Horovod 4
#horovodrun -np 4 -H localhost:4 python train_hvd.py

# horovod with fp16
#horovodrun -np 2 -H localhost:2 python src/train_h.py

# dist train
# export CUDA_VISIBLE_DEVICES=3
# python src/train.py -n 2 -nr 0 2>&1 & > train.log
# export CUDA_VISIBLE_DEVICES=2
# python src/train.py -n 2 -nr 1 2>&1 & > rank1.log

#python gan/train_dcgan.py