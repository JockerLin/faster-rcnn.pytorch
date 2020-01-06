echo "start detect"
pwd
python demo.py --net vgg16 --checksession 1 --checkepoch 20 --checkpoint 10021 --cuda --load_dir models/vgg16/pascal_voc
