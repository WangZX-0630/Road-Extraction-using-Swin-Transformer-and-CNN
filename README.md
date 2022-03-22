##SwinRes UperNet

###参数与运行设置

-设置model_name为SwinRes_UperNet则可选择使用Swin Transformer与ResNet组成的联合backbone

-设置model_name为Swin_UperNet则单独选择Swin Transformer作为backbone

*若需使用多卡分布式训练，则在终端使用命令行*

```
python -m torch.distributed.launch --nproc_per_node {节点gpu数} train.py
```

###数据集使用

*数据存放格式*

```
dataset
    └─────img_dir
    │       └─────train
    │       └─────val
    └─────ann_dir
    │       └─────train
    │       └─────val
    └─────splits
            └─────train.txt
            └─────val.txt
```


####datasets文件夹
-可按照data_SpaceNet.py生成相应的数据集data代码，默认在splits文件夹内有数据集划分的文件

-可以通过修改代码选择就收全部训练集或验证集

-数据集预处理流程可参考mmseg_transforms.py内代码进行组合，sample字典中seg_fileds用来指导label的范围（若需要多张label时）

-imgRGB_mean_std.py可帮助生成RGB图像的mean和std，用以在预处理阶段进行normalize操作


