import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os,cv2,random,json
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def trajectory_strategy(sample,trajectory_dic):
    a = list(range(len(trajectory_dic)))
    random.shuffle(a)
    return a[:sample]
def angle_between_vectors(asart, aend, bstart, bend):
    # 解析坐标
    [x1, y1] = asart
    [x2, y2] = aend
    [x3, y3] = bstart
    [x4, y4] = bend

    # 计算向量的坐标
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    tx = x3 - x1
    ty = y3 - y1
    # 计算夹角的正弦值
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    sin_theta = cross_product / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # 计算夹角
    angle = np.arcsin(sin_theta) * 180 / np.pi  # 转换为角度

    # 根据点积判断夹角正负
    if dot_product < 0:
        if cross_product < 0:
            angle = -180 - angle
        else:
            angle = 180 - angle

    return angle,tx,ty
def Bg_Select(Mask,Bgshape):
    maskshape = Mask.shape
    Bgnumber = [num for num,i in enumerate(Bgshape) if (Mask.shape[0]==i[0] and Mask.shape[1]==i[1])]
    return Bgnumber[0]
def pad_to_5_rows_2_cols(array):
    # 确保输入数组是二维数组
    array = np.atleast_2d(array)
    
    # 获取当前数组的行数
    current_rows = array.shape[0]
    
    # 计算需要填充的行数
    rows_to_add = 5 - current_rows
    
    if rows_to_add > 0:
        # 使用 numpy.pad 填充数组
        padded_array = np.pad(array, ((0, rows_to_add), (0, 0)), 'constant', constant_values=0)
    else:
        padded_array = array

    return padded_array
def txt_generator(a,b,imgpath,imgname):
    h,w,_ = cv2.imread(imgpath).shape
    with open(imgpath.split('.')[0]+'.txt', 'w', encoding='utf-8') as f:
        for fish_box,fish_points in zip(b,a):
            yolo_str = ''
            yolo_str += '{} '.format(0)  #类别
            bbox_top_left_x,bbox_top_left_y,bbox_bottom_right_x,bbox_bottom_right_y = fish_box
            bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
            bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
            # 框宽度
            bbox_width = bbox_bottom_right_x - bbox_top_left_x
            # 框高度
            bbox_height = bbox_bottom_right_y - bbox_top_left_y
            # 框中心点归一化坐标
            bbox_center_x_norm = bbox_center_x / w
            bbox_center_y_norm = bbox_center_y / h
            # 框归一化宽度
            bbox_width_norm = bbox_width / w
            # 框归一化高度
            bbox_height_norm = bbox_height / h
            yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)
            padded_array = pad_to_5_rows_2_cols(fish_points)
            for [px,py] in padded_array:
                if px!=0 and py!=0:
                    keypoint_x_norm = px / w
                    keypoint_y_norm = py / h
                    yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2) # 2-可见不遮挡 1-遮挡 0-没有点
                else:
                    yolo_str += '{:.5f} {:.5f} {} '.format(0, 0, 0)
            f.write(yolo_str + '\n')
def edges_process(DstFish0):
    image = (DstFish0 * 255).astype(np.uint8)
    edges = cv2.Canny(image, 140, 210)

     # 检查 edges 是否包含非零元素
    if np.count_nonzero(edges) == 0:
        print("Error: No non-zero elements in edges!")
        return DstFish0

    blurred_foreground = np.zeros_like(DstFish0)

    # 检查 DstFish0 是否为空
    if DstFish0 is None:
        print("Error: DstFish0 is empty!")
        return DstFish0
    
    blurred_foreground[edges != 0] = cv2.GaussianBlur(DstFish0[edges != 0], (1, 1), 2)
    
    DstFish0[blurred_foreground!=0] = 0
    DstFish0[DstFish0<0.05] = 0
    DstFish0[ DstFish0!=0 ] -= 0.08
    return DstFish0
def resize_and_map_mask_and_keypoints(mask, keypoints, Bg):   
    # 调整前景掩码的尺寸
    resized_mask = cv2.resize(mask, (Bg.shape[1], Bg.shape[0]),interpolation=cv2.INTER_NEAREST)

    # 将关节点的坐标从(300,300)维度映射到(400,500)维度上
    target_width, target_height = Bg.shape[1], Bg.shape[0]
    resized_keypoints = []
    for kp in keypoints:
        # 计算宽度和高度比例
        width_ratio = target_width / mask.shape[1]
        height_ratio = target_height / mask.shape[0]

        # 将关节点的坐标按比例缩放
        mapped_x = int(kp[0] * width_ratio)
        mapped_y = int(kp[1] * height_ratio)
        resized_keypoints.append([mapped_x, mapped_y])

    return resized_mask, np.array(resized_keypoints)
def PaddingMask(mask, keypoints, Bg):
    # 寻找前景掩码的边界框
    nonzero_indices = np.nonzero(mask[:,:,0])
    min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])  #x
    min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])  #y

    foreground_roi = mask[min_row:max_row, min_col:max_col]
    # print(foreground_roi.shape)
    # print(padded_background.shape)

    w = max_col - min_col
    h = max_row - min_row
    # 计算背景图像中心位置
    bg_h, bg_w, _ = Bg.shape
    # print(Bg.shape,mask.shape)

    center_x = (bg_w - w) / 2
    center_y = (bg_h - h) / 2

    # 创建目标大小的图像
    padded_background = np.zeros_like(Bg)

    # 将前景掩码放置在背景图像中心位置
    padded_background[int(center_y):int(center_y)+h, int(center_x):int(center_x)+w] = foreground_roi


    # 计算填充位置
    start_x = (mask.shape[1] - w) / 2
    start_y = (mask.shape[0] - h) / 2


    resized_keypoints = []
    for kp in keypoints:
        resized_keypoints.append([kp[0]+center_x-start_x, kp[1]+center_y-start_y]) #[kp[0]+center_x-start_x, kp[1]+center_y-start_y]
    
    return padded_background, np.array(resized_keypoints)
def draw_rectangle(DstFish,edgepixel):   ##输入：旋转平移后的前景   输出：x1,y1,x2,y2
    nonzero_indices = np.nonzero(DstFish[:,:,0])
    min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])  #x
    min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])  #y
    if min_col-edgepixel >= 0 : 
        min_col-=edgepixel
    if min_row - edgepixel >= 0:
        min_row-=edgepixel
    max_col += edgepixel
    max_row += edgepixel
    width = max_col - min_col
    height = max_row - min_row
    rect = plt.Rectangle((min_col, min_row), width, height, linewidth=1, edgecolor='g', facecolor='none')
    # 添加矩形到图像上
    plt.gca().add_patch(rect)
    return min_col,min_row,max_col,max_row
def draw_points(transformed_point):
    colors = ['r', 'g', 'b', 'c', 'm']
    x_coords = [point[0] for point in transformed_point]
    y_coords = [point[1] for point in transformed_point]
    # 绘制每个点
    for i in range(len(transformed_point)):
        plt.scatter(x_coords[i], y_coords[i], color=colors[i], label=f'Point {i+1}',s=30)
def draw_points2(transformed_point):
    colors = ['white', 'white', 'white', 'white', 'white']
    x_coords = [point[0] for point in transformed_point]
    y_coords = [point[1] for point in transformed_point]
    # 绘制每个点
    for i in range(5):
        plt.scatter(x_coords[i], y_coords[i], color=colors[i], label=f'Point {i+1}',s=5)
def center_point(point,Origin):
    image_size = Origin.shape

    min_row, max_row = np.min( np.squeeze(point[:,1:]) ), np.max( np.squeeze(point[:,1:]) )
    min_col, max_col = np.min(np.squeeze(point[:,:-1])), np.max( np.squeeze(point[:,:-1]) )

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # 计算将掩码移动到图像中心所需的平移量
    center_row = image_size[0] / 2
    center_col = image_size[1] / 2
    delta_row = int(center_row - (min_row + max_row) / 2)
    delta_col = int(center_col - (min_col + max_col) / 2)
    # print('center_point',delta_row,delta_col)
    center_point = [[x+delta_col,y+delta_row] for x,y in point]

    # x_coords = [point[0] for point in center_point]
    # y_coords = [point[1] for point in center_point]
    # # 绘制每个点
    # for i in range(5):
    #     plt.scatter(x_coords[i], y_coords[i], color=colors[i], label=f'Point {i+1}',s=30)

    # plt.imshow(original_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return center_point,[delta_row,delta_col]  
def MoveRotateScaleFish(normalnizefish,point,tx=0,ty=0,angle=0,scale=1):    # 50,50,90,1
    img = np.copy(normalnizefish)  # 假设normalnizefish0是原始图像
    h, w, c = img.shape
    center_x = w / 2
    center_y = h / 2

    homogeneous_point = np.vstack((point.T, np.ones((1, point.shape[0]))))

    # print(homogeneous_point.shape)

    # 构建平移矩阵
    M_translate = np.float32([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]])
    # 构建旋转矩阵
    M_rotate = cv2.getRotationMatrix2D((center_x, center_y), angle, scale)
    M_rotate = np.vstack([M_rotate, [0, 0, 1]])

    # 将平移矩阵和旋转矩阵合并
    M_combined = M_translate.dot(M_rotate)
    # print(M_combined)
    # print(M_combined[:2, :])

    # 进行仿射变换
    dst = cv2.warpAffine(img, M_combined[:2, :], (w, h))
    # 应用仿射变换
    transformed_point = np.dot(M_combined[:2, :], homogeneous_point).T[:, :2]

    # plt.imshow(dst, cmap='gray')
    # plt.title('Shifted fish image')
    # plt.axis('off')

    # plt.show()
    return dst,transformed_point
def mask_fish_MoveCenter(image_size,mask_triangle,fishimgae,type=np.float32):
    nonzero_indices = np.nonzero(mask_triangle)
    # print(nonzero_indices)
    min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    # print(nonzero_indices[0])
    
    # print(min_row, max_row,min_col, max_col)
    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # 计算将掩码移动到图像中心所需的平移量
    center_row = image_size[0] // 2
    center_col = image_size[1] // 2
    delta_row = center_row - (min_row + max_row) // 2
    delta_col = center_col - (min_col + max_col) // 2

    # 创建新的掩码数组，并将三角形掩码移动到图像中心
    new_mask_triangle = np.zeros(image_size, dtype=type)
    for i in range(3):
        new_mask_triangle[min_row + delta_row : min_row + delta_row + height,
                        min_col + delta_col : min_col + delta_col + width,i] = fishimgae[min_row : max_row + 1, min_col : max_col + 1,0]
    return new_mask_triangle
def Generate_SingleNormalizeFish(fish_mask,original_image,move_pixels):

    # print(fish_mask.shape)
    [delta_row,delta_col] = move_pixels

    image_size = original_image.shape
    image_fish = np.copy(original_image)
    image_fish[fish_mask != 1] = 0

    nonzero_indices = np.nonzero(fish_mask)
    # print(nonzero_indices)
    min_row, max_row = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    min_col, max_col = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    # 计算将掩码移动到图像中心所需的平移量
    # delta_row = int(image_size[0]/2 - (min_row + max_row) / 2)
    # delta_col = int(image_size[1]/2 - (min_col + max_col) / 2)
    # print('Generate_SingleNormalizeFish',delta_row,delta_col)
    # print('a',image_size[0]/2,(min_row + max_row) / 2)
    # print('b,',image_size[1]/2 - (min_col + max_col) / 2)

    normalnizefish = np.zeros(original_image.shape, dtype=np.float32)
    for i in range(3):
        normalnizefish[min_row + delta_row : min_row + delta_row + height,
                        min_col + delta_col : min_col + delta_col + width,i] = image_fish[min_row : max_row + 1, min_col : max_col + 1,i]
    return normalnizefish
def Generate_SingleFrame(normalnizefish,points,Bg,matrix1,matrix2):  # 输入:仅前景的掩码图片    原始图     输出:归一化前景图

    x1,y1,delat1,scale1 = matrix1
    x2,y2,delat2,scale2 = matrix2
    
    # -------------------------------------------------------------------------------
    normalnizefishCompute0,pointsCompute0 = PaddingMask(normalnizefish[0],points[0],Bg)
    normalnizefishCompute1,pointsCompute1 = PaddingMask(normalnizefish[1],points[1],Bg)

    normalnizefishCompute0[ normalnizefishCompute0!=0 ] += 0.1
    normalnizefishCompute1[ normalnizefishCompute1!=0 ] += 0.1
    # draw_points(center_point0)
    # draw_points(center_point1)

    DstFish0,transformed_point0 = MoveRotateScaleFish(normalnizefishCompute0,np.array(pointsCompute0),x1,y1,delat1,scale1)    
    DstFish1,transformed_point1 = MoveRotateScaleFish(normalnizefishCompute1,np.array(pointsCompute1),x2,y2,delat2,scale2) 

    edges_process(DstFish0)
    edges_process(DstFish1)

    stenghthen_image = np.maximum(DstFish0,DstFish1)    #两张图片叠加 同时添加背景
    black_mask = (stenghthen_image[:, :, 0] == 0) & (stenghthen_image[:, :, 1] == 0) & (stenghthen_image[:, :, 2] == 0)
    stenghthen_image[black_mask] = Bg[black_mask]
    return stenghthen_image,transformed_point0,transformed_point1,DstFish0,DstFish1
def iou(mask,rect,height,width):   # 计算 框框内部，掩码点的个数
    # 创建一个大小与原始图像相同的空白图像
    blank_image = np.zeros((height,width), dtype=np.uint8)
    # 将 mask 填充到空白图像中
    cv2.fillPoly(blank_image, [mask], 255)
    # 创建一个空白图像，大小与原始图像相同
    rect_mask = np.zeros_like(blank_image)
    # 将矩形填充到空白图像中
    cv2.rectangle(rect_mask, tuple(rect[0]), tuple(rect[1]), 255, -1)
    # 将 mask 和矩形的逻辑与操作，获取矩形内的 mask 区域
    masked_area = cv2.bitwise_and(blank_image, rect_mask)
    # 计算矩形内的 mask 区域面积
    mask_area = cv2.countNonZero(masked_area)
    return mask_area
def pipeibox2mask_point(recs,points,contours,height,width):   #矩形1 对应 点+轮廓  矩形2 对应 点+轮廓
    box1 = np.array(recs[0]).astype(int)    #   0,0,0  1,1,1        #0,1,0  1,0,1  
    bbox1 = [[box1[0][0],box1[0][1]],[box1[2][0],box1[2][1]]]

    
    num1 = 0
    num2 = 0
    for i in range(len(points[0])):
        if cv2.pointPolygonTest(box1, points[0][i], False)==1:
            num1+=1
 
    for i in range(len(points[1])):
        if cv2.pointPolygonTest(box1, points[1][i], False)==1:
            num2+=1

    contournum1 = iou(contours[0],bbox1,height,width)
    contournum2 = iou(contours[1],bbox1,height,width)
    # for i in range(len(contours[0])):
    #     if cv2.pointPolygonTest(box1, contours[0][i].tolist(), False)==1:
    #         contournum1+=1

    # for i in range(len(contours[1])):
    #     if cv2.pointPolygonTest(box1, contours[1][i].tolist(), False)==1:
    #         contournum2+=1
    
    # print(num1,num2,contournum1,contournum2)

    bool_list = [False,num1<num2,contournum1<contournum2]
    number1 =  [0 if not item else 1 for item in bool_list]
    number2 =  [0 if item else 1 for item in bool_list]
    return number1,number2
def getpoints(labelme):    #将 1，2，3，4，5，1，2，3，4，1，2，3 这样的列表 拆分 [1,2,3,4,5] [1,2,3,4] [1,2,3]
    sublist1 = [] 
    points = []
    temppoint = []
    for i in labelme['shapes']:
        if i['label'].isdigit():
        
            if not sublist1 or int(i['label']) == sublist1[-1] + 1:

                sublist1.append(int(i['label']))
                temppoint.append([int(i['points'][0][0]),int(i['points'][0][1])])

            else:
                points.append(temppoint)
                temppoint = []
                temppoint.append([int(i['points'][0][0]),int(i['points'][0][1])])
                sublist1 = [int(i['label'])]
    if sublist1:
        points.append(temppoint)
    return points   #[2,5,2] 两组
def Generate_Tragetory_Frame(image_fish,Pfish,Bg,trajectory_dic,trajectory_number,i):

    he,wi,_= Bg.shape

    fish_point1 = np.array(Pfish[0])
    fish_point2 =  np.array(Pfish[1])

    imageFishMask1 = np.copy(image_fish[0]) 
    imageFishMask2 = np.copy(image_fish[1]) 
    # print(111,fish_point1)
    P1fish1 = [fish_point1[0][0]/wi,1-fish_point1[0][1]/he]
    P2fish1 = [fish_point1[1][0]/wi,1-fish_point1[1][1]/he]

    P1fish2 = [fish_point2[0][0]/wi,1-fish_point2[0][1]/he]
    P2fish2 = [fish_point2[1][0]/wi,1-fish_point2[1][1]/he]

    # print('trajectory_number',trajectory_number)
    # print(i,len(trajectory_dic[trajectory_number]['male']))

    imageFishMask1[ imageFishMask1!=0 ] += 0.08
    imageFishMask2[ imageFishMask2!=0 ] += 0.08

    angle,tx,ty = angle_between_vectors(P2fish1,P1fish1,trajectory_dic[trajectory_number]['male'][i],trajectory_dic[trajectory_number]['male'][i+1])
    Dst1,transformed_point1 = MoveRotateScaleFish(imageFishMask1,np.array(fish_point1),tx=tx*wi,ty=-ty*he,angle=angle)
    angle,tx,ty = angle_between_vectors(P2fish2,P1fish2,trajectory_dic[trajectory_number]['female'][i],trajectory_dic[trajectory_number]['female'][i+1])
    Dst2,transformed_point2 = MoveRotateScaleFish(imageFishMask2,np.array(fish_point2),tx=tx*wi,ty=-ty*he,angle=angle)


    edges_process(Dst1)
    edges_process(Dst2)

    stenghthen_image = np.maximum(Dst1,Dst2)    #两张图片叠加 同时添加背景
    black_mask = (stenghthen_image[:, :, 0] == 0) & (stenghthen_image[:, :, 1] == 0) & (stenghthen_image[:, :, 2] == 0)
    stenghthen_image[black_mask] = Bg[black_mask]

    return stenghthen_image,transformed_point1,transformed_point2,Dst1,Dst2


class data_aug():
    def __init__(self,object_dir):
        # file dir config
        self.Fish ={}

        self.object_dir = object_dir
        self.Bgpath = os.path.join('dataset',object_dir,'aug','BackGround')
        self.BackGround  = os.listdir(self.Bgpath)
        self.Bgshape = [mpimg.imread(os.path.join(self.Bgpath,i)).shape for i in self.BackGround]
        
        
        self.output_path =  os.path.join('dataset',object_dir,'aug','output_images')
        self.visual_path = os.path.join('dataset',object_dir,'aug','visual')
        # file dir config
        self.trajectory_dic = []
        trajectory_path = os.path.join('dataset',object_dir,'aug','trajectory.pkl')
        with open(trajectory_path, 'rb') as f:
            loaded_data = pickle.load(f)
        for message in loaded_data:
            trajectory ={'frame_dir':[],'img_shape':[],'male':[],'female':[]}
            trajectory['frame_dir'] = message['frame_dir']
            trajectory['img_shape'] = message['img_shape']
            h,w = message['img_shape']
            for point in message['keypoint'][0][:,2:3,:]:
                if (point[0][0] != 0 and point[0][1] != 0):
                    x = point[0][0]/w
                    y = point[0][1]/h
                    trajectory['male'].append([x,y])

            for point in message['keypoint'][1][:,2:3,:]:
                if (point[0][0] != 0 and point[0][1] != 0):
                    x = point[0][0]/w
                    y = point[0][1]/h
                    trajectory['female'].append([x,y])
            self.trajectory_dic.append(trajectory)
        # get center mask        
        imgpath = os.path.join('dataset',object_dir,'aug','input')
        jsonlabels = [i for i in os.listdir(imgpath) if i.split('.')[1]=='json']
        Imgs = [i for i in os.listdir(imgpath) if i.split('.')[1]==('png' or 'jpeg' or 'bmp')]
        for num in range(len(Imgs)):   # 从0开始
            original_image = mpimg.imread(os.path.join(imgpath,Imgs[num]))
            self.process_single_json(os.path.join(imgpath,jsonlabels[num]),original_image,num)
       
    def process_single_json(self,labelme_path,original_image,num):
    
        with open(labelme_path, 'r', encoding='utf-8') as f:
            labelme = json.load(f)
        

        height,width =  labelme['imageHeight'],labelme['imageWidth']
        MASK_Point = [ann['points'] for ann in labelme['shapes'] if ann['shape_type']=='polygon']
        recs = [ann['points'] for ann in labelme['shapes'] if ann['shape_type']=='rectangle']

        contours = [np.array(x,dtype=np.int32) for x in MASK_Point]
        points = getpoints(labelme)
        number1,number2 = pipeibox2mask_point(recs,points,contours,height,width)
        self.Fish[num] = []

        x, y, w, h = cv2.boundingRect(contours[number1[2]])
        maskImage = np.zeros((height,width), dtype=np.uint8)
        # 遍历图片所有坐标
        for i in range(x, x + w):
            for j in range(y, y + h):
                if cv2.pointPolygonTest(contours[number1[2]], (i, j), False) > 0:   # 返回外部-1 ，内部1 ，边缘 0
                    maskImage[j, i] = 1

        centerpoint,[delta_row,delta_col] = center_point(np.array(points[number1[1]]),original_image)
        normalnizefish = Generate_SingleNormalizeFish(maskImage,original_image,[delta_row,delta_col])[:,:,:3]
        
        self.Fish[num].append({'points':centerpoint,'mask':normalnizefish})

        x, y, w, h = cv2.boundingRect(contours[number2[2]])
        maskImage = np.zeros((height,width), dtype=np.uint8)
        # 遍历图片所有坐标
        for i in range(x, x + w):
            for j in range(y, y + h):
                if cv2.pointPolygonTest(contours[number2[2]], (i, j), False) > 0:   # 返回外部-1 ，内部1 ，边缘 0
                    maskImage[j, i] = 1

        centerpoint,[delta_row,delta_col] = center_point(np.array(points[number2[1]]),original_image)
        normalnizefish = Generate_SingleNormalizeFish(maskImage,original_image,[delta_row,delta_col])[:,:,:3]
        self.Fish[num].append({'points':centerpoint,'mask':normalnizefish})
   
    def VisualizeAndGenerator_result(self,ijnumber,saveVisual=True,saveYOLO=True,saveLabelme=True,num_sample=4,xy_std_dev=0.2,xy_wave=20,delta=90,delta_wave=60,method='intersection',mean=0):  # i Visualize which image 
        normalnizefish = [self.Fish[ijnumber][0]['mask'],self.Fish[ijnumber][1]['mask']]
        points = [self.Fish[ijnumber][0]['points'],self.Fish[ijnumber][1]['points']]

        plt.figure(figsize=(num_sample*11, num_sample*4))
        ax = num_sample
        num_samples =num_sample**2

        result = ''
        for i in range(num_samples):
            Bgnumber = Bg_Select(normalnizefish[0],self.Bgshape)
            Bg = mpimg.imread(os.path.join(self.Bgpath,self.BackGround[Bgnumber]))[:, :, :3]
            h, w, c = Bg.shape
            # Bg = np.ones(Bg.shape)
            if method =='intersection':
                # 交叉重叠方式生成
                x1 = np.random.randn(1) * (w/2)*xy_std_dev + mean
                y1 = np.random.randn(1) * (h/2)*xy_std_dev + mean

                x2 = x1 + np.random.randn(1) * xy_wave
                y2 = y1 + np.random.randn(1) * xy_wave

                delat1 = np.random.randn(1) * delta + mean
                delat2 = delat1 + np.random.randn(1) * delta_wave
                result = 'intersection'
            else:
                x1 = np.random.randn(1) * (w/2)*xy_std_dev + mean
                y1 = np.random.randn(1) * (h/2)*xy_std_dev + mean

                x2 = np.random.randn(1) * (w/2)*xy_std_dev + mean
                y2 = np.random.randn(1) * (h/2)*xy_std_dev + mean

                delat1 = np.random.randn(1) * delta + mean
                delat2 = np.random.randn(1) * delta + mean
                result = 'norm'

            matrix1 = [x1[0],y1[0],delat1[0],1]  #尺度缩放为1
            matrix2 = [x2[0],y2[0],delat2[0],1]

            SingleFrame,transformed_point0,transformed_point1,DstFish0,DstFish1   = Generate_SingleFrame(normalnizefish,points,Bg,matrix1,matrix2)
            if np.max(SingleFrame)>=1.0: 
                SingleFrame[SingleFrame>=1.0]=1  
            if np.min(SingleFrame)<0:
                SingleFrame[SingleFrame<0]=0  

            outimg = os.path.join(self.output_path,self.object_dir+str(ijnumber)+result+'_{}.png'.format(i+1))
            imagename = self.object_dir+str(ijnumber)+result+'_{}.png'.format(i+1)
            plt.imsave(outimg, SingleFrame)
            plt.subplot(ax, ax, i+1)   
            plt.imshow(SingleFrame) 
            plt.axis('off')

            rec0 = draw_rectangle(DstFish0,2)
            rec1 = draw_rectangle(DstFish1,2)
            if saveYOLO:
                txt_generator([transformed_point0,transformed_point1],[rec0,rec1],outimg,imagename)
            # if saveLabelme:
            #     custom.AugmentSingle_to_json([transformed_point0,transformed_point1],[rec0,rec1],outimg,imagename)     #offer in the future
            
            draw_points(transformed_point0)
            draw_points(transformed_point1)
        if saveVisual:
            plt.savefig(os.path.join(self.visual_path,result+str(ijnumber)+'.tif'),dpi=300) 

    def trajectory_Generator_result(self,ijnumber,num_sample=5,step=2): # step 轨迹中需要跳过的帧数，减少生成量 
        
        list_tragetory = trajectory_strategy(num_sample,self.trajectory_dic)
        normalnizefish = [self.Fish[ijnumber][0]['mask'],self.Fish[ijnumber][1]['mask']]
        points = [self.Fish[ijnumber][0]['points'],self.Fish[ijnumber][1]['points']]

        for i in range(num_sample):
            trajectory_number = list_tragetory[i]
            for j in range(0,len(self.trajectory_dic[trajectory_number]['male'])-1,step): 

                Bgnumber = Bg_Select(normalnizefish[0],self.Bgshape)
                Bg = mpimg.imread(os.path.join(self.Bgpath,self.BackGround[Bgnumber]))[:, :, :3]
                SingleFrame,transformed_point0,transformed_point1,DstFish0,DstFish1  = Generate_Tragetory_Frame(normalnizefish,points,Bg,self.trajectory_dic,trajectory_number,j)

                if np.max(SingleFrame)>=1.0:  
                    SingleFrame[SingleFrame>=1.0]=1  
                if np.min(SingleFrame)<0:
                    SingleFrame[SingleFrame<0]=0  

                transformed_point = []
                rec =[]
                #添加矩形框
                if np.max(DstFish0) != 0:
                    rec0 = draw_rectangle(DstFish0,2)
                    transformed_point.append(transformed_point0)
                    rec.append(rec0)
                if np.max(DstFish1) != 0:
                    rec1 = draw_rectangle(DstFish1,2)
                    transformed_point.append(transformed_point1)
                    rec.append(rec1)

                outimg = os.path.join(self.output_path,self.object_dir+str(ijnumber)+'_traj{}_{}.png'.format(i+1,j+1))
                imagename = self.object_dir+str(ijnumber)+'_traj{}_{}.png'.format(i+1,j+1)
                try:
                    plt.imsave(outimg, SingleFrame)
                    draw_rectangle(DstFish0,2)
                    draw_rectangle(DstFish1,2)
                    draw_points(transformed_point0)
                    draw_points(transformed_point1)
                    txt_generator(transformed_point,rec,outimg,imagename)
                except:
                    continue
                #custom.AugmentSingle_to_json(transformed_point,rec,outimg,imagename)


if __name__ == "__main__":
    # file dir config
    augment_module = data_aug('Fish')
    #Visualize and write image result
    for i in range(len(augment_module.Fish)):
        augment_module.VisualizeAndGenerator_result(i,num_sample=2,method='intersection') # num_sample: generate num_sample*num_sample images by one image    method:intersection normal
        augment_module.VisualizeAndGenerator_result(i,num_sample=2,method='normal')
        augment_module.trajectory_Generator_result(i,num_sample=2)   # num_sample: numbers of trajectory per image
    # plt.savefig('Non-intersection.tif',dpi=300) 
    # plt.show()





