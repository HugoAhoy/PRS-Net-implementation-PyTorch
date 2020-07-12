# 预处理
预处理中`pcl_mesh_sampling.exe -n_samples 1000` 并不会产生1000个点,而是少于1000个点且数量不同,
有些obj只能产生2,3个点的点云,很奇怪,要处理

# Dataset如何getitem
如果getitem方法里返回的是一个dict,那么对于在DataLoader 载入batch时会将按照关键词组装tensor

# NotImplementedError
多半是因为继承下来的方法没有实现,比如我这里写完__init__()就直接在初始化函数里定义forward了.网络中对于输入调用__call__()调用forward(),找不到forward方法,从而抛出异常

# Jupyter 刷新
修改自己的模块后,jupyter需要kernel->restart重启一下才能更新.不然跑出来的结果没变化

# tensor cpu 与 gpu
DataLoader从Dataset中拿出来的数据tensor一般就是CPU上的(根据自己写个getitem而定),如果网络指定了gpu,那么要用`x.cuda()`方法把tensor x放到gpu上

# torch调试
torchsnooper非常好用, 只要在调试函数上加一个装饰器torchsnooper.snoop()就可以看到每一步的tensor变换