#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

#ARCH=$1
#GPUS=$1
#OUT_PATH=$3
#PORT=${PORT:-29500}

python show.py -m net_0_epoch
python show.py -m net_10_epoch
python show.py -m net_20_epoch
python show.py -m net_30_epoch
python show.py -m net_40_epoch
python show.py -m net_50_epoch
python show.py -m net_60_epoch
python show.py -m net_70_epoch
python show.py -m net_80_epoch
python show.py -m net_90_epoch
python show.py -m net_100_epoch
python show.py -m net_110_epoch
python show.py -m net_120_epoch
python show.py -m net_130_epoch
python show.py -m net_140_epoch
python show.py -m net_150_epoch
