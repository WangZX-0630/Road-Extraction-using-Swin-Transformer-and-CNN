import numpy as np
import shapefile
import networkx as nx
import time


file = shapefile.Reader('file/test/test.shp')
bbox = file.bbox
# left_top_x = bbox[]
print(bbox)
img_width = 65536
img_height = 65536
unit_x = (bbox[2] - bbox[0]) / img_width
unit_y = (bbox[3] - bbox[1]) / img_height


def read_shp():
    G = nx.read_shp('file/test/test.shp')
    nx.write_edgelist(G, "file/test.graph")
    f = open('file/test_gt.csv', 'w')
    f.write("ImageId,WKT_Pix\n")
    for line in open("file/test.graph", 'r'):
        LineString = line.split("'Wkt': ")[1].split(", 'Json'")[0]
        # print(LineString)
        figures = LineString.split('(')[1].split(')')[0]
        # print(figures)
        wkt_string = ''
        for point in figures.split(','):
            pixel_x = np.round((float(point.split(' ')[0]) - bbox[0]) / unit_x)
            pixel_y = np.round((float(point.split(' ')[1]) - bbox[1]) / unit_y)
            wkt_string += str(pixel_x) + ' ' + str(pixel_y) + ', '
        wkt_string = wkt_string.strip(', ')
        f.write("AOI_0_test_img0, \"LINESTRING (" + wkt_string + ")\"\n")
    f.close()

def read_geojson():
    f_p = open('file/test_proposal.csv', 'w')
    f_p.write("ImageId,WKT_Pix\n")
    Lines = open('file/test.geojson', 'r').read()
    for line in Lines.split("\"coordinates\": [[")[1:]:
        # print(line)
        line = line.split(']]}')[0]
        wkt_string = ''
        negative_flag = False
        for point in line.split('], ['):
            # print(point)
            pixel_x = np.round((float(point.split(', ')[0]) - bbox[0]) / unit_x)
            pixel_y = np.round((float(point.split(' ')[1]) - bbox[1]) / unit_y)
            if pixel_x >= 0 and pixel_y >= 0:
                wkt_string += str(pixel_x) + ' ' + str(pixel_y) + ', '
            else:
                negative_flag = True
        if not negative_flag:
            wkt_string = wkt_string.strip(', ')
            f_p.write("AOI_0_test_img0, \"LINESTRING (" + wkt_string + ")\"\n")
    f_p.close()


if __name__ == '__main__':
    start_time = time.perf_counter()
    read_shp()
    read_geojson()
    end_time = time.perf_counter()

    print('cost time: %f s' % (end_time - start_time))

    # cost time: 6.94s
