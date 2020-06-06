#! /usr/bin/env bash

alias python3="/usr/bin/python3"
cd data
mkdir processed
python3 preprocessing/data_maker.py
./sum_data.sh