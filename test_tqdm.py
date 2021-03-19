#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/03/19
"""
from tqdm import tqdm

if __name__ == '__main__':
    a = [i for i in range(10)]
    test_tqdm = tqdm(a, desc='test:')
    for index, n in enumerate(test_tqdm):
        print(index, n)