import time

import numpy as np
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2



# f = open('file/test_cut_proposal.csv', 'w')
# f.write('ImageId,WKT_Pix\n')
# for line in open('file/test_proposal.csv', 'r'):
#     if not line.startswith('ImageId'):
#         points = line.split('(')[1].split(')')[0]
#         for i, point in enumerate(points.split(', ')[1:]):
#             f.write('AOI_0_test_img0, "LINESTRING (' + points.split(', ')[i - 1] + ', ' + points.split(', ')[i] + ')\n')
# f.close()


img_width = 65536
img_height = 65536
clip_nums_x = 4
clip_nums_y = 4
unit_x = int(img_width / clip_nums_x)
unit_y = int(img_height / clip_nums_y)


def two_points_cross_area(point1_x, point1_y, point2_x, point2_y, f):
    if point2_x < point1_x:
        two_points_cross_area(point2_x, point2_y, point1_x, point1_y, f)
    # 输入点需要 point2_x > point1_x
    else:
        point1_area = (int(point1_x / unit_x), int(point1_y / unit_y))
        point2_area = (int(point2_x / unit_x), int(point2_y / unit_y))
        # 若两个点在同一个区域内，则直接转化坐标并打印
        if point1_area[0] == point2_area[0] and point1_area[1] == point2_area[1]:
            f.write('AOI_0_test_' + str(int(point1_x / unit_x)) + '_' +
                    str(int(point1_y / unit_y)) + ', "LINESTRING (' + str(int(point1_x - point1_area[0] * unit_x)) +
                    ' ' + str(int(point1_y - point1_area[1] * unit_y)) + ', ' + str(
                int(point2_x - point2_area[0] * unit_x)) + ' ' +
                    str(int(point2_y - point2_area[1] * unit_y)) + ')"\n')
        # 若两个点并非在同一区域，则需找到两点之间连线与两点所在区域边缘的交点
        else:
            slope = (point2_y - point1_y) / (point2_x - point1_x)  # 该条直线的斜率
            # 若该直线在一个point的子图区域内与横轴相交，则交点为（x，0或子图高度），若与纵轴相交，则交点为（0或子图宽度，y）
            # 在point1区域内，需要寻找交点作为第二个点一同送入递归，即可打印出这两点

            if int((slope * ((point1_area[0] + 1) * unit_x - point1_x) + point1_y) / unit_y) == point1_area[1]:
                point3_x = (point1_area[0] + 1) * unit_x - 1
                point3_y = int(slope * ((point1_area[0] + 1) * unit_x - 1 - point1_x) + point1_y)
                point1_flag = 1  # 从point1出发的直线从右侧的区域纵轴穿过
            else:
                if slope > 0:
                    point3_x = int(((point1_area[1] + 1) * unit_y - 1 - point1_y) / slope + point1_x)
                    point3_y = (point1_area[1] + 1) * unit_y - 1
                    point1_flag = 2  # 从point1出发的直线从区域顶部的横轴穿过
                else:
                    point3_x = int(((point1_area[1]) * unit_y - point1_y) / slope + point1_x)
                    point3_y = (point1_area[1]) * unit_y
                    point1_flag = 3  # 从point1出发的直线从区域底部的横轴穿过
            two_points_cross_area(point1_x, point1_y, point3_x, point3_y, f)
            # 在point2区域内，需要寻找交点作为第一个点一同送入递归
            if int((point2_y - slope * (point2_x - point2_area[0] * unit_x)) / unit_y) == point2_area[1]:
                point4_x = (point2_area[0]) * unit_x
                point4_y = int(point2_y - slope * (point2_x - (point2_area[0]) * unit_x))
                point2_flag = 1  # 到达point2的直线从区域左侧的纵轴穿过
            else:
                if slope > 0:
                    point4_x = int(point2_x - (point2_y - (point2_area[1]) * unit_y) / slope)
                    point4_y = point2_area[1] * unit_y
                    point2_flag = 2  # 到达point2的直线从区域底部的横轴穿过
                else:
                    point4_x = int(point2_x - (point2_y - (point2_area[1] + 1) * unit_y) / slope)
                    point4_y = (point2_area[1] + 1) * unit_y - 1
                    point2_flag = 3  # 到达point2的直线从区域顶部的横轴穿过
            two_points_cross_area(point4_x, point4_y, point2_x, point2_y, f)
            # 若point1与point2之间并非只有一层的距离，则交点point3与point4之间仍有区域的空间间隔，递归循环即可
            if not ((point1_area[0] == point2_area[0] and abs(point1_area[1] - point2_area[1]) == 1) or
                    (point2_area[0] - point1_area[0] == 1 and point1_area[1] == point2_area[1])):
                # point3, point4分别与point1 point2在同一区间 需找到其在边界另一侧的对应点 该点与原点不在同一区域
                if point1_flag == 1:
                    point3_x += 1
                    point3_y += int(slope)
                elif point1_flag == 2:
                    point3_y += 1
                    point3_x += int(1 / slope)
                elif point1_flag == 3:
                    point3_y -= 1
                    point3_x -= int(1 / slope)

                if point2_flag == 1:
                    point4_x -= 1
                    point3_y -= int(slope)
                elif point2_flag == 2:
                    point4_y -= 1
                    point4_x -= int(1 / slope)
                elif point2_flag == 3:
                    point4_y += 1
                    point4_x += int(1 / slope)

                two_points_cross_area(point3_x, point3_y, point4_x, point4_y, f)


def cut_big_image(origin_file_path, target_file_path):
    f = open(target_file_path, 'w')
    f.write('ImageId,WKT_Pix\n')
    for line in open(origin_file_path, 'r'):
        if not line.startswith('ImageId'):
            points = line.split('(')[1].split(')')[0]
            point1 = points.split(', ')[0]
            point2 = points.split(', ')[1]
            point1_x = float(point1.split(' ')[0])
            point1_y = float(point1.split(' ')[1])
            point2_x = float(point2.split(' ')[0])
            point2_y = float(point2.split(' ')[1])
            point1_area = (int(point1_x / unit_x), int(point1_y / unit_y))
            point2_area = (int(point2_x / unit_x), int(point2_y / unit_y))
            if int(point1_x / unit_x) == int(point2_x / unit_x) and int(point1_y / unit_y) == int(point2_y / unit_y):
                f.write('AOI_0_test_' + str(int(point1_x / unit_x)) + '_' +
                        str(int(point1_y / unit_y)) + ', "LINESTRING (' + str(int(point1_x - point1_area[0] * unit_x)) +
                        ' ' + str(int(point1_y - point1_area[1] * unit_y)) + ', ' + str(int(point2_x - point2_area[0] * unit_x)) + ' ' +
                        str(int(point2_y - point2_area[1] * unit_y)) + ')"\n')
            else:
                two_points_cross_area(point1_x, point1_y, point2_x, point2_y, f)


def separate_small_imgs(origin_file_path, target_dir, nums_x, nums_y):
    for i in range(nums_x):
        for j in range(nums_y):
            with open(target_dir + 'test_' + str(i) + '_' + str(j) + '.csv', 'w') as f:
                f.write('ImageId,WKT_Pix\n')
    for line in open(origin_file_path, 'r'):
        if not line.startswith('Image'):
            img_id = line.split(',')[0]
            i = img_id.split('_')[3]
            j = img_id.split('_')[4]
            if int(i) >= nums_x or int(j) >= nums_y:
                continue
            with open(target_dir + 'test_' + str(i) + '_' + str(j) + '.csv', 'a+') as f:
                if line not in f.readlines():
                    f.write(line)
                else:
                    continue

    for file in os.listdir(target_dir):
        f = open(target_dir + file, 'r')
        coordinates = set()
        lines = ['ImageId,WKT_Pix\n']
        for line in f.readlines():
            if not line.startswith('Image'):
                coordinate = line.split('LINESTRING ')[1].split('"')[0]
                if coordinate not in coordinates:
                    coordinates.add(coordinate)
                    lines.append(line)
        f = open(target_dir + file, 'w')
        for lin in lines:
            f.write(lin)




if __name__ == '__main__':
    start_time = time.perf_counter()
    cut_big_image('file/test_gt.csv', 'file/test_cut_gt.csv')
    separate_small_imgs('file/test_cut_gt.csv', 'file/test_gt_wkt/', 4, 4)
    end_time = time.perf_counter()
    print('cost time: %f s' % (end_time - start_time))

    # cut_big_image 65536 * 1 -> 16384 * 16: 0.112s(gt)
    # separate_small_imgs 16384 * 16: 3.987s(gt), 0.433s(proposal)