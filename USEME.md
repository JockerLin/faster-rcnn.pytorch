训练：

vgg 迭代次数为20

```shell
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16 --bs 1 --nw 1 --lr 0.001 --lr_decay_step 5 --cuda
```



测试：

暂时因为gpu内存问题测试不了，一次性读入太多img

```shell
python test_net.py --dataset pascal_voc --net vgg16 --checksession 1 --checkepoch 20 --checkpoint 10021 --cuda
```




检测images文件夹下的图片
```shell
python demo.py --net vgg16 --checksession 1 --checkepoch 20 --checkpoint 10021 --cuda --load_dir models/vgg16/pascal_voc
```

