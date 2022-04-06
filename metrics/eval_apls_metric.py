#!/user/bin/python
# coding=utf-8

from skimage.morphology import thin
from multiprocessing import Pool

import sys
import os
import shutil
import time
import numpy as np
import cv2 as cv
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name", type=str, help="file_name to name result.csv", default="test_batch_apls"
)
parser.add_argument(
    "--wkt_dir", type=str, help="input predict wkt dir", default="C:/Users/Lenovo/PycharmProjects/VecRoad-master/data/input/proposal_batch"
)
parser.add_argument(
    "--gt_dir", type=str, help="input gt wkt dir", default="C:/Users/Lenovo/PycharmProjects/VecRoad-master/data/input/gt_batch"
)
parser.add_argument(
    "--save_dir", type=str, help="save result.csv dir", default="C:/Users/Lenovo/PycharmProjects/VecRoad-master/data/graphs/"
)
parser.add_argument(
    "--apls_path", type=str, help="apls metric jar dir", default="C:/Users/Lenovo/PycharmProjects/VecRoad-master/data/downloads/apls_metric.jar"
)

args = parser.parse_args()

files = os.listdir(args.gt_dir)
files.sort()

def worker(fn):
    fn = fn.split('.')[0]
    ret = os.popen(
            "java -jar {} -truth {}/{}.csv -solution {}/{}.csv -no-gui"
            .format(args.apls_path, args.gt_dir, fn, args.wkt_dir, fn)
        ).readlines()
    print(ret)
    return ret

if __name__ == '__main__':
    start_time = time.perf_counter()
    res_lst = []
    # files = ['denver.graph']
    # pool = Pool(8)
    tmp_lst = []
    ret_lst = []
    for fn in files:
        # tmp_lst.append(pool.apply_async(worker, args=(fn,)))
        tmp_lst.append(worker(fn))
    for item in tmp_lst:
        ret_lst.append(item)
    # pool.close()
    # pool.join()
    end_time = time.perf_counter()


    res_file = open(os.path.join(args.save_dir, '{}.csv'.format(args.file_name)), 'w')
    log_file = open(os.path.join(args.save_dir, '{}.log'.format(args.file_name)), 'w')
    for ret in ret_lst:
        for line in ret.get():
            print()
            log_file.write(line)
        log_file.write('\n')
        res_lst.append(float(ret.get()[-1].strip().split(' ')[-1]))
    for i in range(len(files)):
        fn = files[i]
        res = res_lst[i]
        res_file.write("{},{}\n".format(fn, res))
    res_file.write("{},{}".format(args.file_name, sum(res_lst)/len(res_lst)))
    res_file.close()
    log_file.close()

    print('cost time: %f s' % (end_time - start_time))

    # 16384 * 16: 13.785s
    # 1024 * 8 batch: 0.246s