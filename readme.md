# [pytorch 多GPU训练](https://blog.csdn.net/minstyrain/article/details/127731963?spm=1001.2014.3001.5501)

1. 普通单机单卡训练流程，以mnist为例 [train.py](train.py)

2. DDP分布式训练 [train_ddp.py](train_ddp.py)
```
首先计算出当前进程序号：rank = args.nr * args.gpus + gpu，
然后就是通过dist.init_process_group初始化分布式环境，其中backend参数指定通信后端，包括mpi, gloo, nccl，这里选择nccl，这是Nvidia提供的官方多卡通信框架，相对比较高效。
mpi也是高性能计算常用的通信协议，不过你需要自己安装MPI实现框架，比如OpenMPI。gloo倒是内置通信后端，但是不够高效。init_method指的是如何初始化，以完成刚开始的进程同步；
这里我们设置的是env://，指的是环境变量初始化方式，需要在环境变量中配置4个参数：MASTER_PORT，MASTER_ADDR，WORLD_SIZE，RANK，前面两个参数我们已经配置，后面两个参数也可以通过dist.init_process_group函数中world_size和rank参数配置
```

3. horovod分布式训练 [train_horovod.py](train_horovod.py)
```
添加hvd.init()来初始化Horovod；
为每个worker分配GPU，一般一个worker process对应一个GPU，对应关系通过rank id来映射。例如pytorch中为torch.cuda.set_device(hvd.local_rank())
随着world_size的变化，batch_size也在变化，因此我们也要随着world_size的变化来调整lr，一般为原有的lr 值乘以world_size；
将原有深度学习框架的optimizer通过horovod中的hvd.DistributedOptimizer进行封装；
rank 0 将初始的variable广播给所有worker: hvd.broadcast_parameters(model.state_dict(), root_rank=0)
仅在worker 0上进行checkpoint的save
```
运行方式, 双卡为例
```
horovodrun -np 2 -H localhost:2 python train_horovod.py
```