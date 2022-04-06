import sys
import os
import numpy as np
import time
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


def get_line_string_dict(path):
    linestrings = {}
    reader = open(path, 'r')
    for line in reader:
        img_id = line.split(',')[0]
        l = []
        if line == 'ImageId,WKT_Pix\n':
            continue
        else:
            figures = line.split('(')[1].split(')')[0]
            figures = figures.split(', ')

            for f in range(0, len(figures)):
                k = figures[f].split(' ')
                l += [[int(float(k[0])), int(float(k[1]))]]
        if img_id in linestrings:
            linestrings[img_id] += [l]
        else:
            linestrings[img_id] = [l]
    return linestrings


def draw_small_imgs(nums_x, nums_y, whole_img_width, whole_img_height, file_path):
    linestrings = get_line_string_dict(file_path)
    for i in range(nums_x):
        for j in range(nums_y):
            seg_gt = np.zeros((whole_img_width / nums_x, whole_img_height / nums_y), dtype=np.float)
            img_name = 'AOI_0_test_' + str(i) + '_' + str(j)
            lines = linestrings[img_name]
            if lines[0] is not None:
                for l in lines:
                    for points in range(1, len(l)):
                        cv2.line(seg_gt, (l[points][0], l[points][1]), (l[points - 1][0], l[points - 1][1]), 255,
                                 thickness=1)
            cv2.imwrite('file/test' + str(i) + '_' + str(j) + '.png', seg_gt)


def draw_whole_img(whole_img_width, whole_img_height, file_path, img_name, target_path):
    linestrings = get_line_string_dict(file_path)
    seg_gt = np.zeros((whole_img_width, whole_img_height), dtype=np.float)
    img_name = img_name
    lines = linestrings[img_name]
    if lines[0] is not None:
        for l in lines:
            for points in range(1, len(l)):
                cv2.line(seg_gt, (l[points][0], l[points][1]), (l[points - 1][0], l[points - 1][1]), 255,
                         thickness=1)
    cv2.imwrite(target_path + img_name + '.png', seg_gt)


def cut_big_img_physically(nums_x, nums_y, whole_img_width, whole_img_height):
    unit_x = int(whole_img_width / nums_x)
    unit_y = int(whole_img_height / nums_y)
    img = cv2.imread('file/test_proposal.png', cv2.IMREAD_GRAYSCALE)
    for i in range(nums_x):
        for j in range(nums_y):
            image = img[i*unit_x:i*unit_x + unit_x, j*unit_x:j*unit_y + unit_y]
            cv2.imwrite('file/test_mask/test_' + str(j) + '_' + str(i) + '.png', image)


if __name__ == '__main__':
    start_time = time.perf_counter()
    # draw_small_imgs(4, 4, 65536, 65536, 'file/test_cut_gt.csv')
    draw_whole_img(1024, 1024, 'csv/100129_proposal.csv', '100129', 'csv/proposal_')
    draw_whole_img(1024, 1024, 'csv/100129_gt.csv', '100129', 'csv/gt_')
    # cut_big_img_physically(4, 4, 65536, 65536)
    end_time = time.perf_counter()
    print('cost time: %f s' % (end_time - start_time))

