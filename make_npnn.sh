#!/usr/bin/env bash

if [ "$1" == '' ] || [ "$2" == '' ]
then
   echo "Usage: $0 cfg_file npnn_file"
   exit
fi

#python3 test.py --cfg experiments/cfgs/ssd_lite_mbv2_train_np_rb_total_flag.yml --onnx=/tmp/rb_flag_new_anchor.onnx
python3 test.py --cfg=$1 --onnx=$2.onnx

../ncnn/build-host-gcc-linux/tools/onnx/onnx2ncnn $2.onnx  $2.param  $2.bin
cat $2.onnx.npnn.header $2.param $2.bin > $2

