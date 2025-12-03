from . import FPsort
from . import custom
from ultralytics import YOLO
import cv2,json,os
import numpy as np
import lap
# import FPsort
# import custom



def get_first_item(my_dict):
   
    for key, value in my_dict.items():
        return key, value
def infer(img,model):
    result = model(source=img,imgsz=96,verbose= False)[0]
    # print(result.probs)
    # print(result.names)
    conf = result.probs.top1conf.item()
    cls = result.names[result.probs.top1]

    return cls,conf

def distance_batch(pp_test,pp_gt):   #(x1-x2)**2+(y1-y2)**2+....(xn-xn)**2
    try:
      result = np.linalg.norm(pp_test[:, np.newaxis, :] - pp_gt, axis=2)
    except:
      print(pp_test,pp_gt)
      return []
    return result  #欧式距离

def calculate_iou(box1, box2):
    # box1 和 box2 的格式为 ((x1, y1), (x2, y2), confidence)
    x1a, y1a, x1b, y1b = box1
    x2a, y2a,x2b, y2b = box2
    
    # 计算交集的坐标
    x_inter = max(0, min(x1b, x2b) - max(x1a, x2a))
    y_inter = max(0, min(y1b, y2b) - max(y1a, y2a))
    
    # 计算交集的面积
    intersection_area = x_inter * y_inter
    
    # 计算各自框的面积
    area_box1 = (x1b - x1a) * (y1b - y1a)
    area_box2 = (x2b - x2a) * (y2b - y2a)
    
    # 计算并集的面积
    union_area = area_box1 + area_box2 - intersection_area
    
    # 计算交并比
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou
def Anomaly_detection(count,frame_current,frame_last,history_Points,current_Points,history_boxes,current_boxes,fix_num=2):  #使用该算法 删除其他关键点   ------------- 未弄


    check_points=[]
    truth_keypoints = []
    truth_boxes = []

    if len(frame_current)>fix_num:
        cost_matrix = distance_batch(np.array(frame_current),np.array(frame_last))
        
        value, x, y = lap.lapjv(cost_matrix, extend_cost=True)

        # print('检测到斑马鱼个数：',len(frame_current))
        # print('当前帧数',count)
        # print(cost_matrix,value, x, y)
        # print(frame_current,frame_last)

        check_points = [frame_current[i] for i in y]
        truth_keypoints = [current_Points[i] for i in y]
        truth_boxes = [current_boxes[i] for i in y]

    elif  0<len(frame_current)<fix_num:

        cost_matrix = distance_batch(np.array(frame_current),np.array(frame_last))
    
        value, x, y = lap.lapjv(cost_matrix, extend_cost=True)

  
        for num,i in enumerate(y):
            if i==-1:
                check_points.append(frame_last[num])
                truth_keypoints.append(history_Points[num])
                truth_boxes.append(history_boxes[num])
            else:
                check_points.append(frame_current[i])
                truth_keypoints.append(current_Points[i])
                truth_boxes.append(current_boxes[i])


    elif len(frame_current) == 0:
    #    print('检测到斑马鱼个数： 0')
    #    print('当前帧数',count)
    #    print(frame_current,frame_last)
       check_points = frame_last 
       truth_keypoints = history_Points
       truth_boxes =  history_boxes
    else:
        check_points = frame_current
        truth_keypoints = current_Points
        truth_boxes = current_boxes
    return check_points,truth_keypoints,truth_boxes
def InputReader(path):
    cap=cv2.VideoCapture("{}".format(path))  #视频流读取
    rate = cap.get(5)   # 帧速率
    FrameNumber = cap.get(7)  # 视频文件的帧数
    duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
    w = cap.get(3)   # 宽
    h = cap.get(4)  # 高
    return cap,rate,FrameNumber,duration,w,h

mot_tracker = FPsort.FPSort(max_age=10, 
                min_hits=10) #create instance of the SORT tracker    删除参数，创建新的目标参数

class Muti_vedio_output():

    def __init__(self,pose_model,tracking_method,reid_model,Anormaly_dection,inputdir,projectname,save_vedio=True,save_json=True,show=True):
        self.model =  YOLO(pose_model)  
      
        if reid_model['status']:
            self.model2 = YOLO(reid_model['model_PATH'])
        # self.capture = capture
        # self.vedioname = os.path.basename(capture)
        self.Anormaly_dection = Anormaly_dection
        self.tracking_method = tracking_method
        self.resultdir = os.path.join('results',projectname)
        self.project = projectname
        self.inputpath = inputdir

        self.save_vedio=save_vedio
        self.save_json=save_json
        self.show=show

        self.inti_ID_connect = {}
        self.middle_ID_connect = {}
        self.all_ID_connect = {}
        self.record_OverlapORIntersection = {}
        self.OverlapORIntersection_FLAG =False

    def _inference(self,capture,write=False,show=False,json_write=False,Noneactoinframe=3):
        jsondict = {'name':0,'frames':0,'total_fames':0,'keypoints':{},'boxes':{}}
        cap,rate,FrameNumber,duration,w,h = InputReader(capture)
        frameidx=0
        trackers = []
        ID_connect = None
        KeepID_connect = None
        flaga =False
        npy = []
        history_boxes=0
        history_Points=0
        vedioname = os.path.basename(capture).split('.')[0]

        if write:
            # rate = int(cap.get(5))
            out = cv2.VideoWriter(os.path.join(self.resultdir,self.tracking_method+'_'+vedioname+'.mp4'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(rate), (int(w), int(h)))    #(int(frame_width*0.8), int(frame_height*0.8)

        while cap.isOpened():
 
            rec,img = cap.read()
            frameidx+=1

            if rec == False:  
                break
            # img = cv2.convertScaleAbs(img, alpha=0.8, beta=30)

            result = self.model(img, imgsz=640, conf=0.5,verbose=False)[0]
            if result.keypoints.conf != None:
            # print(result.keypoints)
                points = [points[2:3].tolist()[0] for points in result.keypoints.data] #获取中心点
                current_Points = result.keypoints.data.cpu().numpy() #获取全部点
                current_boxes = result.boxes.data[:,:-1].cpu().numpy()
                if len(current_Points) == 2 and flaga==False:  #进行初始化
                    history_Points = current_Points  
                    history_boxes = current_boxes
                    flaga = True
                if flaga ==False:
                    continue
                if len(trackers)!=0 and flaga == True:
                    tempp = [[x[4],x[5],x[10]] for x in trackers]
                    points,history_Points,history_boxes = Anomaly_detection(frameidx,points,tempp,history_Points,current_Points,history_boxes,current_boxes) 

                combined_arr = np.stack(history_Points, axis=0)
                tp = combined_arr.reshape((len(history_boxes),5,3))[:,:,:-1].reshape((len(history_boxes),10))
         
                trackers,matchID = mot_tracker.update(tp) 
                # print( np.array(points))
                if frameidx>2:
                    ID_connect = self.distance_close(matchID,frameidx,img,self.model2,result, np.array(points),threshold=25)  # {'male': 1, 'female': 0}  需要调节阈值  很重要
                    if ID_connect!=None:   #保持状态
                        KeepID_connect = ID_connect
                # print(KeepID_connect)

                # ---------- json numpy存储 ----------------------
                if json_write:
                    if frameidx > Noneactoinframe: 
                        # print(frameidx)
                        try:
                            if KeepID_connect == None:
                                KeepID_connect = {'male':0,'female':1}
                            PointsID = [x[1] for x in matchID]
                            ID_male = PointsID.index(KeepID_connect['male'])
                        except:
                            # print(frameidx)
                            print('KeepID_connect,matchID :')
                            print(KeepID_connect,matchID )
                        ID_female = PointsID.index(KeepID_connect['female'])

                        temppoints = np.copy(history_Points)
                        temppoints[:,:,0] =temppoints[:,:,0]/w
                        temppoints[:,:,1] =temppoints[:,:,1]/h

                        npy.append(np.hstack((temppoints[ID_male].flatten(),temppoints[ID_female].flatten())))  # 先male 后female

                        if frameidx%(rate*60) == 0:
                            np.save(os.path.join(self.resultdir,self.tracking_method+'_'+vedioname+'.npy'),np.array(npy))
                            npy = []
                        
                        jsondict['keypoints'][frameidx] = {'male':history_Points[ID_male].tolist(),'female':history_Points[ID_female].tolist()}
                        jsondict['boxes'][frameidx] = {'male':history_boxes[ID_male].tolist(),'female':history_boxes[ID_female].tolist()}
                # ---------- -------- ----------------------

            custom.tracker_point_visiualize(img,result,trackers,KeepID_connect)
            cv2.putText(img, str(frameidx), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 2)

            if write:
                out.write(img)  #视频写入           
            if show:
                cv2.namedWindow("enhanced",0)
                cv2.resizeWindow("enhanced", 700, 300)
                cv2.imshow('enhanced',img)

            del img
            if cv2.waitKey(1)==ord('q'):
                break
        
        cap.release() 
        if write:
            out.release()
        if json_write:
            jsondict['name'] = vedioname
            jsondict['frames'] = rate
            jsondict['total_fames'] = FrameNumber
            with open(os.path.join(self.resultdir,self.tracking_method+'_'+vedioname+'.json'), "w") as json_file:   #修改存储路径
                json.dump(jsondict, json_file, indent=1, separators=(',', ':'))
        if show:
            cv2.destroyAllWindows()
    
    def muti_inference(self):
        resultdir = os.path.join('results',self.project)
        try:
            os.makedirs(resultdir)
        except:
            print('create filedir already')
        for vedioname in os.listdir(self.inputpath):
            capture = os.path.join(self.inputpath,vedioname)
            self._inference(capture,write=self.save_vedio,show=self.show,json_write=self.save_json)
        
    def distance_close(self,matchID,frameidx,img,model2,result,points,threshold=50):   #这个阈值非常重要
       
        pics = result.boxes.xyxy.cpu().numpy()
        #初始判断过程
        if 2<frameidx<=200 and len(result.keypoints)==2:
            if calculate_iou(pics[0],pics[1])<0.15:
                temp = []
                for box in pics:
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    cls,conf = infer(img[y1 : y2, x1: x2],model2)
                    temp.append(cls)
                if len(set(temp)) != 1:
                    self.inti_ID_connect[frameidx] = {temp[0]:matchID[0][1],temp[1]:matchID[1][1]}
                    self.all_ID_connect[frameidx] = {temp[0]:matchID[0][1],temp[1]:matchID[1][1]}  #可以删除 记录错误

            if len(self.inti_ID_connect) == 3: #首先判断是否有4个完全匹配ID与性别 。如果匹配，则确定。 如果不匹配则不确定                                  yaoxiugai
                males = [self.inti_ID_connect[a]['male'] for a in self.inti_ID_connect]   
                females = [self.inti_ID_connect[a]['female'] for a in self.inti_ID_connect]
                if len(set(males))+len(set(females)) == 2:
                    return get_first_item(self.inti_ID_connect)[1]
                else:
                    self.inti_ID_connect = {}
                    return None
            else:
                return None
        #中间判断过程   关节点距离很近小于阈值时 进行记录
        else:
            # tempA = [listtwo[2] for listtwo in result.keypoints.cpu().numpy()]  #看使用哪个点跟踪的   这里修改
            if np.linalg.norm(points[0][:2]-points[1][:2])<=threshold:
                self.record_OverlapORIntersection[frameidx]  = points
                self.OverlapORIntersection_FLAG =True
                # print('find overlap or Interserction!!')

            if len(result.keypoints)==2:
                if calculate_iou(pics[0],pics[1]) <= 0.15 and self.OverlapORIntersection_FLAG==True:
                    tempB = []
                    for box in pics:
                        x1 = int(box[0])
                        y1 = int(box[1])
                        x2 = int(box[2])
                        y2 = int(box[3])
                        cls,conf = infer(img[y1 : y2, x1: x2],model2)
                        tempB.append(cls)
                    if len(set(tempB)) != 1:
                        self.middle_ID_connect[frameidx] = {tempB[0]:matchID[0][1],tempB[1]:matchID[1][1]}
                        self.all_ID_connect[frameidx] = {tempB[0]:matchID[0][1],tempB[1]:matchID[1][1]}  #可以删除 记录错误
                        # print(frameidx,middle_ID_connect)

            if len(self.middle_ID_connect) == 5: #首先判断是否有4个完全匹配ID与性别 。如果匹配，则确定。 如果不匹配则不确定
                males = [self.middle_ID_connect[a]['male'] for a in self.middle_ID_connect]   
                females = [self.middle_ID_connect[a]['female'] for a in self.middle_ID_connect]
                if len(set(males))+len(set(females)) == 2:
                    OverlapORIntersection_FLAG = False
                    temp_result = get_first_item(self.middle_ID_connect)[1]
                    self.middle_ID_connect = {}
                    return temp_result
                else:
                    self.middle_ID_connect = {}
                    return None
            else:
                return None

if __name__ == "__main__":
    PATH = 'D:/user/SZ/Project/DLC/vedio/' #'./predict/data/vedio/'
    resultdir = os.path.join('results','Zebrash_tracking')
    a = Muti_vedio_output('models/best_hand.pt','FPsort',{'status': True,'model_PATH':'models/gender.pt'},{'status': True, 'fixnum': 2},PATH,resultdir,'Zebrash_tracking')
    a.muti_inference()





        
        


