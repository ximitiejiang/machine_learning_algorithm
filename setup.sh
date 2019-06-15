#!/usr/bin/env bash

pip3 install pandas
pip3 install numpy
pip3 install sklearn

mkdir dataset
cd dataset
mkdir mnist
wget -c https://github.com/WenDesi/lihang_book_algorithm/blob/master/data/train_binary.csv
wget -c https://github.com/WenDesi/lihang_book_algorithm/blob/master/data/train.csv
