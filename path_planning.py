import cv2 as cv
import numpy as np
import sys
import itertools
import math
import cv2

class point:
    id=0 #点的序号
    place_x=0 #点的x坐标
    place_y=0 #点的y坐标

    #初始化
    def __init__(self,place_x, place_y):
        self.id = place_y*10+place_x
        self.place_x = place_x
        self.place_y = place_y

Points=[] #所有点的集合
# 定义每个点的坐标
def init_points():
    for i in range(100):
        Points.append(point(i%10,i//10))

warped_image = None  # 全局变量，用于存储透视变换后的图像

def detect_squares(image_path):
    global side_length
    global warped_image  # 声明为全局变量，以便在函数外部访问

    # 读取图像
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 高斯平滑滤波
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Canny边缘检测
    edges = cv.Canny(blurred, 50, 150)

    # 轮廓检测
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 初始化变量
    center_points = []

    # 迭代所有轮廓
    for contour in contours:
        perimeter = cv.arcLength(contour, True)

        # 使用多边形逼近轮廓
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果逼近多边形具有4个顶点，则被认为是正方形
        if len(approx) == 4:
            # 计算中心点坐标
            M = cv.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_points.append((cX, cY))

   # 对中心点进行排序
    center_points.sort(key=lambda p: (p[0]*p[1]))
    max_x_max_y = center_points[-1]
    min_x_min_y = center_points[0]
    # print(center_points)
    del center_points[0]
    del center_points[-1]
    center_points.sort(key=lambda p: (p[0]))
    # print(center_points)
    min_x_max_y = center_points[0]
    max_x_min_y = center_points[-1]
    # print(min_x_min_y,min_x_max_y,max_x_min_y,max_x_max_y)
    # 定义目标正方形的边长
    side_length = int(math.sqrt((max_x_max_y[0] - min_x_min_y[0]) * (max_x_max_y[0] - min_x_min_y[0])  + (max_x_max_y[1] - min_x_min_y[1]) * (max_x_max_y[1] - min_x_min_y[1])))
    # print(side_length)
    # 定义原图中四个顶点坐标和目标图中四个顶点坐标
    src_pts = np.float32([min_x_min_y, min_x_max_y, max_x_min_y, max_x_max_y])
    dst_pts = np.float32([[0, 0], [0, side_length], [side_length, 0], [side_length, side_length]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换
    warped_image = cv2.warpPerspective(image, M, (side_length, side_length))

    return warped_image

# 图像路径
image_path = r"C:\Users\xxx\Desktop\bisai.png"

# 调用函数进行正方形检测和截取
img = detect_squares(image_path)










gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 中值滤波
img = cv.medianBlur(gray_img, 3)

# 高斯模糊处理
blurred = cv.GaussianBlur(img, (5, 5), 0)

# 圆形检测
circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT_ALT, 1, 5, param1=500, param2=0.8, minRadius=10, maxRadius=50)

# 检查是否检测到圆
if circles is not None:
    circles = np.round(circles[0, :]).astype(int)

    # 绘制检测到的圆
    for (x, y, r) in circles:
        cv.circle(warped_image, (x, y), r, (0, 255, 0), 2)

    # 显示图像
    # cv.imshow('Detected Circles', warped_image)
    # cv.waitKey(0)
else:
    print("未检测到圆")

circles = np.uint16(np.around(circles))
circles_x=[]
circles_y=[]
circles_r=[]
for (x, y, r) in circles:
    circles_x.append(x)
    circles_y.append(y)
    circles_r.append(r)
#把坐标加到列表里
# a, b, c = circles.shape
# print(a, b, c)
# print(str(circles))


final_target = []
for i in range(len(circles_x)):
    x = ((side_length - circles_x[i])*13)/side_length - 0.5
    y = ((side_length - circles_y[i])*13)/side_length - 0.5
    final_target.append((int(x), int(y)))
final_target.sort(key=lambda z: (z[0]+z[1]))
# print(final_target)
treasure_points=[]
for i in range(len(final_target)):
    treasure_points.append(point(10-final_target[i][0],final_target[i][1]-1))
    print(10-final_target[i][0],final_target[i][1]-1)
# cv.imshow('Detected Circles', warped_image)
# cv.waitKey(0)
# draw_img = cv.merge((img.copy(), img.copy(), img.copy()))
# for i in range(len(circles_x)):
#     cv.circle(draw_img, (circles_x[i], circles_y[i]), circles_r[i], (0, 0, 255), 3, cv.LINE_AA)  # 画圆轮廓
    # cv.circle(draw_img, (circles[0][i][0], circles[0][i][1]), 2, (0, 0, 255), 3, cv.LINE_AA)  # 画圆心

# cv.imwrite('/Users/cailingyu/Documents/test_img/point230.png', draw_img)
# cv.imshow("circles", draw_img)
# cv.waitKey(0)
# cv.destroyAllWindows()









# P1=point(0, 0)    #第一个点的坐标为(0,0)
# P2=point(1, 0)    #第二个点的坐标为(1,0)
# P3=point(2, 0)    #第三个点的坐标为(2,0)
# P4=point(3, 0)    #第四个点的坐标为(3,0)
# P5=point(4, 0)    #第五个点的坐标为(4,0)
# P6=point(5, 0)    #第六个点的坐标为(5,0)
# P7=point(6, 0)    #第七个点的坐标为(6,0)
# P8=point(7, 0)    #第八个点的坐标为(7,0)
# P9=point(8, 0)    #第九个点的坐标为(8,0)
# P10=point(9, 0)   #第十个点的坐标为(9,0)
# P11=point(0, 1)  #第十一个点的坐标为(0,1)
# P12=point(1, 1)  #第十二个点的坐标为(1,1)
# P13=point(2, 1)  #第十三个点的坐标为(2,1)
# P14=point(3, 1)  #第十四个点的坐标为(3,1)
# P15=point(4, 1)  #第十五个点的坐标为(4,1)
# P16=point(5, 1)  #第十六个点的坐标为(5,1)
# P17=point(6, 1)  #第十七个点的坐标为(6,1)
# P18=point(7, 1)  #第十八个点的坐标为(7,1)
# P19=point(8, 1)  #第十九个点的坐标为(8,1)
# P20=point(9, 1)  #第二十个点的坐标为(9,1)
# P21=point(0, 2)  #第二十一个点的坐标为(0,2)
# P22=point(1, 2)  #第二十二个点的坐标为(1,2)
# P23=point(2, 2)  #第二十三个点的坐标为(2,2)
# P24=point(3, 2)  #第二十四个点的坐标为(3,2)
# P25=point(4, 2)  #第二十五个点的坐标为(4,2)
# P26=point(5, 2)  #第二十六个点的坐标为(5,2)
# P27=point(6, 2)  #第二十七个点的坐标为(6,2)
# P28=point(7, 2)  #第二十八个点的坐标为(7,2)
# P29=point(8, 2)  #第二十九个点的坐标为(8,2)
# P30=point(9, 2)  #第三十个点的坐标为(9,2)
# P31=point(0, 3)  #第三十一个点的坐标为(0,3)
# P32=point(1, 3)  #第三十二个点的坐标为(1,3)
# P33=point(2, 3)  #第三十三个点的坐标为(2,3)
# P34=point(3, 3)  #第三十四个点的坐标为(3,3)
# P35=point(4, 3)  #第三十五个点的坐标为(4,3)
# P36=point(5, 3)  #第三十六个点的坐标为(5,3)
# P37=point(6, 3)  #第三十七个点的坐标为(6,3)
# P38=point(7, 3)  #第三十八个点的坐标为(7,3)
# P39=point(8, 3)  #第三十九个点的坐标为(8,3)
# P40=point(9, 3)  #第四十个点的坐标为(9,3)
# P41=point(0, 4)  #第四十一个点的坐标为(0,4)
# P42=point(1, 4)  #第四十二个点的坐标为(1,4)
# P43=point(2, 4)  #第四十三个点的坐标为(2,4)
# P44=point(3, 4)  #第四十四个点的坐标为(3,4)
# P45=point(4, 4)  #第四十五个点的坐标为(4,4)
# P46=point(5, 4)  #第四十六个点的坐标为(5,4)
# P47=point(6, 4)  #第四十七个点的坐标为(6,4)
# P48=point(7, 4)  #第四十八个点的坐标为(7,4)
# P49=point(8, 4)  #第四十九个点的坐标为(8,4)
# P50=point(9, 4)  #第五十个点的坐标为(9,4)
# P51=point(0, 5)  #第五十一个点的坐标为(0,5)
# P52=point(1, 5)  #第五十二个点的坐标为(1,5)
# P53=point(2, 5)  #第五十三个点的坐标为(2,5)
# P54=point(3, 5)  #第五十四个点的坐标为(3,5)
# P55=point(4, 5)  #第五十五个点的坐标为(4,5)
# P56=point(5, 5)  #第五十六个点的坐标为(5,5)
# P57=point(6, 5)  #第五十七个点的坐标为(6,5)
# P58=point(7, 5)  #第五十八个点的坐标为(7,5)
# P59=point(8, 5)  #第五十九个点的坐标为(8,5)
# P60=point(9, 5)  #第六十个点的坐标为(9,5)
# P61=point(0, 6)  #第六十一个点的坐标为(0,6)
# P62=point(1, 6)  #第六十二个点的坐标为(1,6)
# P63=point(2, 6)  #第六十三个点的坐标为(2,6)
# P64=point(3, 6)  #第六十四个点的坐标为(3,6)
# P65=point(4, 6)  #第六十五个点的坐标为(4,6)
# P66=point(5, 6)  #第六十六个点的坐标为(5,6)
# P67=point(6, 6)  #第六十七个点的坐标为(6,6)
# P68=point(7, 6)  #第六十八个点的坐标为(7,6)
# P69=point(8, 6)  #第六十九个点的坐标为(8,6)
# P70=point(9, 6)  #第七十个点的坐标为(9,6)
# P71=point(0, 7)  #第七十一个点的坐标为(0,7)
# P72=point(1, 7)  #第七十二个点的坐标为(1,7)
# P73=point(2, 7)  #第七十三个点的坐标为(2,7)
# P74=point(3, 7)  #第七十四个点的坐标为(3,7)
# P75=point(4, 7)  #第七十五个点的坐标为(4,7)
# P76=point(5, 7)  #第七十六个点的坐标为(5,7)
# P77=point(6, 7)  #第七十七个点的坐标为(6,7)
# P78=point(7, 7)  #第七十八个点的坐标为(7,7)
# P79=point(8, 7)  #第七十九个点的坐标为(8,7)
# P80=point(9, 7)  #第八十个点的坐标为(9,7)
# P81=point(0, 8)  #第八十一个点的坐标为(0,8)
# P82=point(1, 8)  #第八十二个点的坐标为(1,8)
# P83=point(2, 8)  #第八十三个点的坐标为(2,8)
# P84=point(3, 8)  #第八十四个点的坐标为(3,8)
# P85=point(4, 8)  #第八十五个点的坐标为(4,8)
# P86=point(5, 8)  #第八十六个点的坐标为(5,8)
# P87=point(6, 8)  #第八十七个点的坐标为(6,8)
# P88=point(7, 8)  #第八十八个点的坐标为(7,8)
# P89=point(8, 8)  #第八十九个点的坐标为(8,8)
# P90=point(9, 8)  #第九十个点的坐标为(9,8)
# P91=point(0, 9)  #第九十一个点的坐标为(0,9)
# P92=point(1, 9)  #第九十二个点的坐标为(1,9)
# P93=point(2, 9)  #第九十三个点的坐标为(2,9)
# P94=point(3, 9)  #第九十四个点的坐标为(3,9)
# P95=point(4, 9)  #第九十五个点的坐标为(4,9)
# P96=point(5, 9)  #第九十六个点的坐标为(5,9)
# P97=point(6, 9)  #第九十七个点的坐标为(6,9)
# P98=point(7, 9)  #第九十八个点的坐标为(7,9)
# P99=point(8, 9)  #第九十九个点的坐标为(8,9)
# P100=point(9, 9) #第一百个点的坐标为(9,9)

start_point= point(0,0)#出发点
out_point= point(9,9)#出口点
# treasure_points=[] #宝藏点
# treasure_points=[point(2,2),point(4,3),point(8,4),point(8,1),point(1,8),point(1,5),point(5,6),point(7,7)] #宝藏点
start_treasure_out_points=[]

#寻找离出发点最近的宝藏点
# def find_nearest_treasure_in_point():
#     min_distance=sys.maxsize
#     for i in range(8):
#         if distance[start_point.id][treasure_points[i].id]<min_distance:
#             min_distance=distance[start_point.id][treasure_points[i].id]
#             nearest_treasure_point=treasure_points[i]
#     return nearest_treasure_point

# #寻找离出口点最近的宝藏点
# def find_nearest_treasure_out_point():
#     min_distance=sys.maxsize
#     for i in range(8):
#         if distance[out_point.id][treasure_points[i].id]<min_distance:
#             min_distance=distance[out_point.id][treasure_points[i].id]
#             nearest_treasure_point=treasure_points[i]
#     return nearest_treasure_point



#出发点和出口点
#def get_start_and_out_point():

    #输入出发点接口
    #start_point_place = get()
    #start_point=polint(start_point_place[0],start_point_place[1])

    #输入出口点接口
    #out_point_place = get()
    #out_point=polint(out_point_place[0],out_point_place[1])

#    pass

#宝藏点
#def get_treasure_point():
    # 读取宝藏点的坐标
    # 。。。。
    #pass

def fill_start_treasure_out_points():
    start_treasure_out_points.append(start_point)
    for i in range(8):
        start_treasure_out_points.append(treasure_points[i])
    start_treasure_out_points.append(out_point)


# 距离矩阵
distance=[[sys.maxsize for i in range(100)] for j in range(100)]

#路径矩阵
floyd_path=[[-1 for i in range(100)] for j in range(100)]

#初始化距离矩阵
def init_distance():
    distance[Points[0].id][Points[0].id]=0
    distance[Points[1].id][Points[1].id]=0
    distance[Points[2].id][Points[2].id]=0
    distance[Points[3].id][Points[3].id]=0
    distance[Points[4].id][Points[4].id]=0
    distance[Points[5].id][Points[5].id]=0
    distance[Points[6].id][Points[6].id]=0
    distance[Points[7].id][Points[7].id]=0
    distance[Points[8].id][Points[8].id]=0
    distance[Points[9].id][Points[9].id]=0
    distance[Points[10].id][Points[10].id]=0
    distance[Points[11].id][Points[11].id]=0
    distance[Points[12].id][Points[12].id]=0
    distance[Points[13].id][Points[13].id]=0
    distance[Points[14].id][Points[14].id]=0
    distance[Points[15].id][Points[15].id]=0
    distance[Points[16].id][Points[16].id]=0
    distance[Points[17].id][Points[17].id]=0
    distance[Points[18].id][Points[18].id]=0
    distance[Points[19].id][Points[19].id]=0
    distance[Points[20].id][Points[20].id]=0
    distance[Points[21].id][Points[21].id]=0
    distance[Points[22].id][Points[22].id]=0
    distance[Points[23].id][Points[23].id]=0
    distance[Points[24].id][Points[24].id]=0
    distance[Points[25].id][Points[25].id]=0
    distance[Points[26].id][Points[26].id]=0
    distance[Points[27].id][Points[27].id]=0
    distance[Points[28].id][Points[28].id]=0
    distance[Points[29].id][Points[29].id]=0
    distance[Points[30].id][Points[30].id]=0
    distance[Points[31].id][Points[31].id]=0
    distance[Points[32].id][Points[32].id]=0
    distance[Points[33].id][Points[33].id]=0
    distance[Points[34].id][Points[34].id]=0
    distance[Points[35].id][Points[35].id]=0
    distance[Points[36].id][Points[36].id]=0
    distance[Points[37].id][Points[37].id]=0
    distance[Points[38].id][Points[38].id]=0
    distance[Points[39].id][Points[39].id]=0
    distance[Points[40].id][Points[40].id]=0
    distance[Points[41].id][Points[41].id]=0
    distance[Points[42].id][Points[42].id]=0
    distance[Points[43].id][Points[43].id]=0
    distance[Points[44].id][Points[44].id]=0
    distance[Points[45].id][Points[45].id]=0
    distance[Points[46].id][Points[46].id]=0
    distance[Points[47].id][Points[47].id]=0
    distance[Points[48].id][Points[48].id]=0
    distance[Points[49].id][Points[49].id]=0
    distance[Points[50].id][Points[50].id]=0
    distance[Points[51].id][Points[51].id]=0
    distance[Points[52].id][Points[52].id]=0
    distance[Points[53].id][Points[53].id]=0
    distance[Points[54].id][Points[54].id]=0
    distance[Points[55].id][Points[55].id]=0
    distance[Points[56].id][Points[56].id]=0
    distance[Points[57].id][Points[57].id]=0
    distance[Points[58].id][Points[58].id]=0
    distance[Points[59].id][Points[59].id]=0
    distance[Points[60].id][Points[60].id]=0
    distance[Points[61].id][Points[61].id]=0
    distance[Points[62].id][Points[62].id]=0
    distance[Points[63].id][Points[63].id]=0
    distance[Points[64].id][Points[64].id]=0
    distance[Points[65].id][Points[65].id]=0
    distance[Points[66].id][Points[66].id]=0
    distance[Points[67].id][Points[67].id]=0
    distance[Points[68].id][Points[68].id]=0
    distance[Points[69].id][Points[69].id]=0
    distance[Points[70].id][Points[70].id]=0
    distance[Points[71].id][Points[71].id]=0
    distance[Points[72].id][Points[72].id]=0
    distance[Points[73].id][Points[73].id]=0
    distance[Points[74].id][Points[74].id]=0
    distance[Points[75].id][Points[75].id]=0
    distance[Points[76].id][Points[76].id]=0
    distance[Points[77].id][Points[77].id]=0
    distance[Points[78].id][Points[78].id]=0
    distance[Points[79].id][Points[79].id]=0
    distance[Points[80].id][Points[80].id]=0
    distance[Points[81].id][Points[81].id]=0
    distance[Points[82].id][Points[82].id]=0
    distance[Points[83].id][Points[83].id]=0
    distance[Points[84].id][Points[84].id]=0
    distance[Points[85].id][Points[85].id]=0
    distance[Points[86].id][Points[86].id]=0
    distance[Points[87].id][Points[87].id]=0
    distance[Points[88].id][Points[88].id]=0
    distance[Points[89].id][Points[89].id]=0
    distance[Points[90].id][Points[90].id]=0
    distance[Points[91].id][Points[91].id]=0
    distance[Points[92].id][Points[92].id]=0
    distance[Points[93].id][Points[93].id]=0
    distance[Points[94].id][Points[94].id]=0
    distance[Points[95].id][Points[95].id]=0
    distance[Points[96].id][Points[96].id]=0
    distance[Points[97].id][Points[97].id]=0
    distance[Points[98].id][Points[98].id]=0
    distance[Points[99].id][Points[99].id]=0
    
    distance[Points[0].id][Points[1].id]=1
    distance[Points[1].id][Points[0].id]=1
    distance[Points[1].id][Points[11].id]=1
    distance[Points[11].id][Points[1].id]=1
    distance[Points[11].id][Points[10].id]=1
    distance[Points[10].id][Points[11].id]=1
    distance[Points[10].id][Points[20].id]=1
    distance[Points[20].id][Points[10].id]=1
    distance[Points[20].id][Points[21].id]=1
    distance[Points[21].id][Points[20].id]=1
    distance[Points[21].id][Points[31].id]=1
    distance[Points[31].id][Points[21].id]=1
    distance[Points[31].id][Points[32].id]=1
    distance[Points[32].id][Points[31].id]=1
    distance[Points[32].id][Points[42].id]=1
    distance[Points[42].id][Points[32].id]=1
    distance[Points[42].id][Points[43].id]=1
    distance[Points[43].id][Points[42].id]=1
    distance[Points[43].id][Points[53].id]=1
    distance[Points[53].id][Points[43].id]=1
    distance[Points[53].id][Points[54].id]=1
    distance[Points[54].id][Points[53].id]=1
    distance[Points[54].id][Points[55].id]=1
    distance[Points[55].id][Points[54].id]=1
    distance[Points[55].id][Points[56].id]=1
    distance[Points[56].id][Points[55].id]=1
    distance[Points[56].id][Points[57].id]=1
    distance[Points[57].id][Points[56].id]=1
    distance[Points[57].id][Points[58].id]=1
    distance[Points[58].id][Points[57].id]=1
    distance[Points[58].id][Points[59].id]=1
    distance[Points[59].id][Points[58].id]=1
    distance[Points[59].id][Points[69].id]=1
    distance[Points[69].id][Points[59].id]=1
    distance[Points[59].id][Points[49].id]=1
    distance[Points[49].id][Points[59].id]=1
    distance[Points[49].id][Points[39].id]=1
    distance[Points[39].id][Points[49].id]=1
    distance[Points[39].id][Points[38].id]=1
    distance[Points[38].id][Points[39].id]=1
    distance[Points[38].id][Points[28].id]=1
    distance[Points[28].id][Points[38].id]=1
    distance[Points[28].id][Points[27].id]=1
    distance[Points[27].id][Points[28].id]=1
    distance[Points[27].id][Points[37].id]=1
    distance[Points[37].id][Points[27].id]=1
    distance[Points[37].id][Points[47].id]=1
    distance[Points[47].id][Points[37].id]=1
    distance[Points[47].id][Points[48].id]=1
    distance[Points[48].id][Points[47].id]=1
    distance[Points[27].id][Points[26].id]=1
    distance[Points[26].id][Points[27].id]=1
    distance[Points[26].id][Points[36].id]=1
    distance[Points[36].id][Points[26].id]=1
    distance[Points[36].id][Points[46].id]=1
    distance[Points[46].id][Points[36].id]=1
    distance[Points[46].id][Points[45].id]=1
    distance[Points[45].id][Points[46].id]=1
    distance[Points[45].id][Points[44].id]=1
    distance[Points[44].id][Points[45].id]=1
    distance[Points[44].id][Points[43].id]=1
    distance[Points[43].id][Points[44].id]=1
    distance[Points[36].id][Points[35].id]=1
    distance[Points[35].id][Points[36].id]=1
    distance[Points[35].id][Points[34].id]=1
    distance[Points[34].id][Points[35].id]=1
    distance[Points[27].id][Points[17].id]=1
    distance[Points[17].id][Points[27].id]=1
    distance[Points[17].id][Points[16].id]=1
    distance[Points[16].id][Points[17].id]=1
    distance[Points[16].id][Points[6].id]=1
    distance[Points[6].id][Points[16].id]=1
    distance[Points[6].id][Points[5].id]=1
    distance[Points[5].id][Points[6].id]=1
    distance[Points[5].id][Points[15].id]=1
    distance[Points[15].id][Points[5].id]=1
    distance[Points[15].id][Points[25].id]=1
    distance[Points[25].id][Points[15].id]=1
    distance[Points[25].id][Points[24].id]=1
    distance[Points[24].id][Points[25].id]=1
    distance[Points[24].id][Points[23].id]=1
    distance[Points[23].id][Points[24].id]=1
    distance[Points[23].id][Points[33].id]=1
    distance[Points[33].id][Points[23].id]=1
    distance[Points[33].id][Points[32].id]=1
    distance[Points[32].id][Points[33].id]=1
    distance[Points[23].id][Points[13].id]=1
    distance[Points[13].id][Points[23].id]=1
    distance[Points[13].id][Points[14].id]=1
    distance[Points[14].id][Points[13].id]=1
    distance[Points[14].id][Points[15].id]=1
    distance[Points[15].id][Points[14].id]=1
    distance[Points[14].id][Points[4].id]=1
    distance[Points[4].id][Points[14].id]=1
    distance[Points[4].id][Points[3].id]=1
    distance[Points[3].id][Points[4].id]=1
    distance[Points[3].id][Points[2].id]=1
    distance[Points[2].id][Points[3].id]=1
    distance[Points[2].id][Points[12].id]=1
    distance[Points[12].id][Points[2].id]=1
    distance[Points[12].id][Points[22].id]=1
    distance[Points[22].id][Points[12].id]=1
    distance[Points[6].id][Points[7].id]=1
    distance[Points[7].id][Points[6].id]=1
    distance[Points[7].id][Points[8].id]=1
    distance[Points[8].id][Points[7].id]=1
    distance[Points[8].id][Points[18].id]=1
    distance[Points[18].id][Points[8].id]=1
    distance[Points[8].id][Points[9].id]=1
    distance[Points[9].id][Points[8].id]=1
    distance[Points[9].id][Points[19].id]=1
    distance[Points[19].id][Points[9].id]=1
    distance[Points[19].id][Points[29].id]=1
    distance[Points[29].id][Points[19].id]=1
    distance[Points[57].id][Points[67].id]=1
    distance[Points[67].id][Points[57].id]=1
    distance[Points[67].id][Points[68].id]=1
    distance[Points[68].id][Points[67].id]=1
    distance[Points[68].id][Points[78].id]=1
    distance[Points[78].id][Points[68].id]=1
    distance[Points[78].id][Points[79].id]=1
    distance[Points[79].id][Points[78].id]=1
    distance[Points[79].id][Points[89].id]=1
    distance[Points[89].id][Points[79].id]=1
    distance[Points[88].id][Points[89].id]=1
    distance[Points[89].id][Points[88].id]=1
    distance[Points[98].id][Points[88].id]=1
    distance[Points[88].id][Points[98].id]=1
    distance[Points[98].id][Points[99].id]=1
    distance[Points[99].id][Points[98].id]=1
    distance[Points[67].id][Points[66].id]=1
    distance[Points[66].id][Points[67].id]=1
    distance[Points[66].id][Points[76].id]=1
    distance[Points[76].id][Points[66].id]=1
    distance[Points[76].id][Points[75].id]=1
    distance[Points[75].id][Points[76].id]=1
    distance[Points[75].id][Points[74].id]=1
    distance[Points[74].id][Points[75].id]=1
    distance[Points[74].id][Points[84].id]=1
    distance[Points[84].id][Points[74].id]=1
    distance[Points[84].id][Points[85].id]=1
    distance[Points[85].id][Points[84].id]=1
    distance[Points[85].id][Points[86].id]=1
    distance[Points[86].id][Points[85].id]=1
    distance[Points[86].id][Points[76].id]=1
    distance[Points[76].id][Points[86].id]=1
    distance[Points[85].id][Points[95].id]=1
    distance[Points[95].id][Points[85].id]=1
    distance[Points[95].id][Points[96].id]=1
    distance[Points[96].id][Points[95].id]=1
    distance[Points[96].id][Points[97].id]=1
    distance[Points[97].id][Points[96].id]=1
    distance[Points[97].id][Points[87].id]=1
    distance[Points[87].id][Points[97].id]=1
    distance[Points[87].id][Points[77].id]=1
    distance[Points[77].id][Points[87].id]=1
    distance[Points[84].id][Points[94].id]=1
    distance[Points[94].id][Points[84].id]=1
    distance[Points[94].id][Points[93].id]=1
    distance[Points[93].id][Points[94].id]=1
    distance[Points[93].id][Points[92].id]=1
    distance[Points[92].id][Points[93].id]=1
    distance[Points[92].id][Points[91].id]=1
    distance[Points[91].id][Points[92].id]=1
    distance[Points[91].id][Points[90].id]=1
    distance[Points[90].id][Points[91].id]=1
    distance[Points[90].id][Points[80].id]=1
    distance[Points[80].id][Points[90].id]=1
    distance[Points[80].id][Points[70].id]=1
    distance[Points[70].id][Points[80].id]=1
    distance[Points[91].id][Points[81].id]=1
    distance[Points[81].id][Points[91].id]=1
    distance[Points[93].id][Points[83].id]=1
    distance[Points[83].id][Points[93].id]=1
    distance[Points[83].id][Points[82].id]=1
    distance[Points[82].id][Points[83].id]=1
    distance[Points[82].id][Points[72].id]=1
    distance[Points[72].id][Points[82].id]=1
    distance[Points[72].id][Points[71].id]=1
    distance[Points[71].id][Points[72].id]=1
    distance[Points[71].id][Points[61].id]=1
    distance[Points[61].id][Points[71].id]=1
    distance[Points[61].id][Points[60].id]=1
    distance[Points[60].id][Points[61].id]=1
    distance[Points[60].id][Points[50].id]=1
    distance[Points[50].id][Points[60].id]=1
    distance[Points[50].id][Points[40].id]=1
    distance[Points[40].id][Points[50].id]=1
    distance[Points[40].id][Points[30].id]=1
    distance[Points[30].id][Points[40].id]=1
    distance[Points[40].id][Points[41].id]=1
    distance[Points[41].id][Points[40].id]=1
    distance[Points[41].id][Points[42].id]=1
    distance[Points[42].id][Points[41].id]=1
    distance[Points[72].id][Points[62].id]=1
    distance[Points[62].id][Points[72].id]=1
    distance[Points[62].id][Points[52].id]=1
    distance[Points[52].id][Points[62].id]=1
    distance[Points[52].id][Points[51].id]=1
    distance[Points[51].id][Points[52].id]=1
    distance[Points[72].id][Points[73].id]=1
    distance[Points[73].id][Points[72].id]=1
    distance[Points[73].id][Points[63].id]=1
    distance[Points[63].id][Points[73].id]=1
    distance[Points[63].id][Points[64].id]=1
    distance[Points[64].id][Points[63].id]=1
    distance[Points[64].id][Points[65].id]=1
    distance[Points[65].id][Points[64].id]=1
    distance[Points[63].id][Points[53].id]=1
    distance[Points[53].id][Points[63].id]=1
#调用Flyod算法计算距离矩阵
def Flyod():
    for k in range(100):
        for i in range(100):
            for j in range(100):
                if distance[i][j]>distance[i][k]+distance[k][j]:
                    distance[i][j]=distance[i][k]+distance[k][j]
                    floyd_path[i][j]=k


start_treasure_distance=[[sys.maxsize for i in range(10)] for i in range(10)]
#得到只有出发点+宝藏点的距离矩阵
def get_start_treasure_distance():
    for i in range(len(start_treasure_out_points)):
        for j in range(len(start_treasure_out_points)):
            start_treasure_distance[i][j]=distance[start_treasure_out_points[i].id][start_treasure_out_points[j].id]
    return(start_treasure_distance)

start_index = 0
end_index = 9
intermediates = [1,2,3,4,5,6,7,8]
# tsp_path=[]
# min_sum_distance=sys.maxsize
# sum_distance=0
# min_sum_distance=sys.maxsize

#使用xiao潇神算法得到最短路径
def xiao():
    numbers = list(range(1, 9))
    permutations = list(itertools.permutations(numbers))

    # 将元组转换为列表
    permutations = [list(permutation) for permutation in permutations]

    #  输出全排列
    min_sum_distance=sys.maxsize
    tsp_path=[]
    for permutation in permutations:
        # print(permutation)
        # tsp_path=[]
        # min_sum_distance=sys.maxsize
        sum_distance=start_treasure_distance[0][permutation[0]]+start_treasure_distance[permutation[7]][9]
        for i in range(7):
            sum_distance=sum_distance+start_treasure_distance[permutation[i]][permutation[i+1]]
        if sum_distance < min_sum_distance:
            min_sum_distance = sum_distance
            tsp_path = permutation
    tsp_path.insert(0,0)
    tsp_path.append(9)
    for i in range(10):
        tsp_path[i]=start_treasure_out_points[tsp_path[i]].id 
    #tsp_path输出的测试口
    # print(min_sum_distance)
    # print(tsp_path)
    return (tsp_path)
tsp_path=[]
real_path=[]


#调用Floyd_path算法得到具体路径
def get_path(s=start_index,e=end_index):
    if floyd_path[s][e]==-1:
        real_path.append(Points[e])
        real_path.insert(0,Points[s])
    else:
        get_path(s,floyd_path[s][e])
        get_path(floyd_path[s][e],e)
    return(real_path)
    # for i in range(len(real_path)):
    #     print(real_path[i].id)
final_path=[]
xxx_path=[]
#得到最终路径
# def get_final_path(tsp_path):
#     final_path=[]
#     temp_path=[]
#     temp=0
#     for i in range(9):
#         final_path=get_path(tsp_path[i],tsp_path[i+1])
#         temp=len(final_path)
#         for j in range(temp//2-1):
#             del final_path[0]
#         del final_path[0]
#         for k in range(len(final_path)):
#             print(final_path[k].id)
#     pass

# 操作
#direction:0 左转 1 直行 2 右转 31 掉头并一格 32 掉头并两格 4 停止 5 空

def operate(last_place_x, last_place_y, now_place_x, now_place_y, next_place_x, next_place_y):
    # 计算当前位置与前一个位置的方向向量
    direction_vector = (now_place_x - last_place_x, now_place_y - last_place_y)
    
    # 计算当前位置与目标位置的方向向量
    target_vector = (next_place_x - now_place_x, next_place_y - now_place_y)
    
    # 判断方向向量的变化情况，确定左转、直行或右转
    if next_place_x==last_place_x and next_place_y==last_place_y:#掉头
        if now_place_x==2 and now_place_y==2:
            return 32 #掉头2
        elif now_place_x==8 and now_place_y==1:
            return 31 #掉头1
        elif now_place_x==9 and now_place_y==2:
            return 32 #掉头2
        elif now_place_x==0 and now_place_y==3:
            return 31 #掉头1
        elif now_place_x==4 and now_place_y==3:
            return 32 #掉头2
        elif now_place_x==8 and now_place_y==4:
            return 31 #掉头1
        elif now_place_x==1 and now_place_y==5:
            return 31 #掉头1
        elif now_place_x==9 and now_place_y==6:
            return 31 #掉头1
        elif now_place_x==5 and now_place_y==6:
            return 32 #掉头2
        elif now_place_x==0 and now_place_y==7:
            return 32 #掉头2
        elif now_place_x==1 and now_place_y==8:
            return 31 #掉头1
        elif now_place_x==7 and now_place_y==7:
            return 32 #掉头2
        
    #判断是否为长直线点
    if now_place_x==3 and now_place_y==0:
        return 5
    elif now_place_x==7 and now_place_y==0:
        return 5
    elif now_place_x==2 and now_place_y==1:
        return 5
    elif now_place_x==9 and now_place_y==1:
        return 5
    elif now_place_x==4 and now_place_y==2:
        return 5
    elif now_place_x==5 and now_place_y==3:
        return 5
    elif now_place_x==7 and now_place_y==3:
        return 5
    elif now_place_x==1 and now_place_y==4:
        return 5
    elif now_place_x==4 and now_place_y==4:
        return 5
    elif now_place_x==5 and now_place_y==4:
        return 5
    elif now_place_x==9 and now_place_y==4:
        return 5
    elif now_place_x==0 and now_place_y==5:
        return 5
    elif now_place_x==4 and now_place_y==5:
        return 5
    elif now_place_x==5 and now_place_y==5:
        return 5
    elif now_place_x==8 and now_place_y==5:
        return 5
    elif now_place_x==2 and now_place_y==6:
        return 5
    elif now_place_x==4 and now_place_y==6:
        return 5
    elif now_place_x==5 and now_place_y==7:
        return 5
    elif now_place_x==0 and now_place_y==8:
        return 5
    elif now_place_x==7 and now_place_y==8:
        return 5
    elif now_place_x==2 and now_place_y==9:
        return 5
    elif now_place_x==6 and now_place_y==9:
        return 5

    elif direction_vector[0] != 0:
        if direction_vector[0] > 0:
            if target_vector[1] > 0:
                return 0  # 右转
            elif target_vector[1] == 0:
                return 1  # 直行
            else:
                return 2  # 左转
        else:
            if target_vector[1] > 0:
                return 2  # 左转
            elif target_vector[1] == 0:
                return 1  # 直行
            else:
                return 0  # 右转
    else:
        if direction_vector[1] > 0:
            if target_vector[0] > 0:
                return 2  # 左转
            elif target_vector[0] == 0:
                return 1  # 直行
            else:
                return 0  # 右转
        else:
            if target_vector[0] > 0:
                return 0  # 右转
            elif target_vector[0] == 0:
                return 1  # 直行
            else:
                return 2  # 左转

directions=[]
# 具体操作
# def get_all_directions():
#     for i in range(1,len(real_path)-1):
#         directions.append(operate(real_path[i-1].place_x,real_path[i-1].place_y,real_path[i].place_x,real_path[i].place_y,real_path[i+1].place_x,real_path[i+1].place_y))
#     # print(directions)

def main():
    #get_start_and_out_point()
    #get_treasure_point()
    init_points()
    fill_start_treasure_out_points()
    init_distance()
    for i in range(len(start_treasure_out_points)):
        if(start_treasure_out_points[i].id==13):
            distance[Points[13].id][Points[14].id]=sys.maxsize
            distance[Points[14].id][Points[13].id]=sys.maxsize
        elif(start_treasure_out_points[i].id==25):
            distance[Points[25].id][Points[15].id]=sys.maxsize
            distance[Points[15].id][Points[25].id]=sys.maxsize
        elif(start_treasure_out_points[i].id==74):
            distance[Points[74].id][Points[84].id]=sys.maxsize
            distance[Points[84].id][Points[74].id]=sys.maxsize
        elif(start_treasure_out_points[i].id==86):
            distance[Points[86].id][Points[76].id]=sys.maxsize
            distance[Points[76].id][Points[86].id]=sys.maxsize
    Flyod()
    start_treasure_distance=get_start_treasure_distance()
    # find_nearest_treasure_in_point()
    # find_nearest_treasure_out_point()
    tsp_path=xiao()
    for i in range(9):
        final_path=get_path(tsp_path[i],tsp_path[i+1])
    for i in range(len(final_path)//2-1,len(final_path)-1):
        xxx_path.append(final_path[i].id)
    xxx_path.append(99)
    

    #测试用接口
    # for i in range(len(xxx_path)):
    #     print(xxx_path[i])
    
    #测试用接口
    # print(len(xxx_path))
    
    #测试用接口
    # i=1
    # directions[i-1]=operate(Points[xxx_path[i-1]].place_x,Points[xxx_path[i-1]].place_y,Points[xxx_path[i]].place_x,Points[xxx_path[i]].place_y,Points[xxx_path[i+1]].place_x,Points[xxx_path[i+1]].place_y)
    # print(directions[i-1])
    
    #测试用接口
    # for i in range(97,98):
    #     print(operate(Points[xxx_path[i-1]].place_x,Points[xxx_path[i-1]].place_y,Points[xxx_path[i]].place_x,Points[xxx_path[i]].place_y,Points[xxx_path[i+1]].place_x,Points[xxx_path[i+1]].place_y))
    # print(len(xxx_path))
    
    #输出具体的路径
    long=len(xxx_path)
    # print(long)
    dir=[]
    final_dir=[]
    for i in range(1,long-1):
        # for i in range(i,i+1):
        dir.append(operate(Points[xxx_path[i-1]].place_x,Points[xxx_path[i-1]].place_y,Points[xxx_path[i]].place_x,Points[xxx_path[i]].place_y,Points[xxx_path[i+1]].place_x,Points[xxx_path[i+1]].place_y))
    for i in range(len(dir)):
        if dir[i]!=5:
            final_dir.append(dir[i])
    print(final_dir)
    #测试用接口
    # for j in range(len(directions)):
    #     print(directions[j])
    
    # print(tsp_path)
    # for i in range(9):
    #     get_path(tsp_path(i),tsp_path(i+1))
    # get_final_path(tsp_path)
    # final_path=[0]
    # for i in range(1,3):
    #     real_path=get_path(tsp_path[i],tsp_path[i+1])
    #     for j in range(len(real_path)//2-1):
    #         del real_path[0]
    #     del real_path[0]
    #     for k in range(len(real_path)):
    #         print(real_path[k].id)
    #     for i in range(len(real_path)):
    #         final_path.append(real_path[i])
    # for i in range(len(final_path)):
    #     print(final_path[i].id)
    
    # get_all_directions()
    pass

if __name__ == '__main__':
    main()

#小熊的第一个自己调试出结果的实际应用代码
#淞淞的耐心指导，框架搭设和熬夜奋斗，祝淞淞夏令营全优营！！！
#感谢蔡姐的图像识别代码

