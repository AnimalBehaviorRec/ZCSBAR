from . import trackconfig
from . import custom
from ultralytics import YOLO
import cv2,json,os
import numpy as np
import lap
# import trackconfig
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

    x1a, y1a, x1b, y1b = box1
    x2a, y2a,x2b, y2b = box2
    x_inter = max(0, min(x1b, x2b) - max(x1a, x2a))
    y_inter = max(0, min(y1b, y2b) - max(y1a, y2a))
    intersection_area = x_inter * y_inter
    area_box1 = (x1b - x1a) * (y1b - y1a)
    area_box2 = (x2b - x2a) * (y2b - y2a)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou
def Anomaly_detection(history_Points,current_Points,history_boxes,current_boxes,fix_num=2):  #使用该算法 删除其他关键点   ------------- 未弄
    truth_keypoints = []
    truth_boxes = []
    combined_arr1 = np.stack(current_Points, axis=0)
    combined_arr2 = np.stack(history_Points, axis=0)
    cp1 = combined_arr1.reshape((len(current_boxes),-1,3))[:,:,:-1].reshape((len(current_boxes),-1))
    hp1 = combined_arr2.reshape((len(history_boxes),-1,3))[:,:,:-1].reshape((len(history_boxes),-1))
   
    if len(cp1)>fix_num:
        cost_matrix = distance_batch(cp1,hp1)
        value, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        # print('检测到斑马鱼个数：',len(frame_current))
        # print('当前帧数',count)
        # print(cost_matrix,value, x, y)
        # print(frame_current,frame_last)
       
        truth_keypoints = [current_Points[i] for i in y]
        truth_boxes = [current_boxes[i] for i in y]
    elif  0<len(cp1)<fix_num:
        cost_matrix = distance_batch(cp1,hp1)
        value, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        for num,i in enumerate(y):
            if i==-1:
                truth_keypoints.append(history_Points[num])
                truth_boxes.append(history_boxes[num])
            else:
                truth_keypoints.append(current_Points[i])
                truth_boxes.append(current_boxes[i])
    elif len(cp1) == 0:
    #    print('检测到斑马鱼个数： 0')
    #    print('当前帧数',count)
    #    print(frame_current,frame_last)
       truth_keypoints = history_Points
       truth_boxes =  history_boxes
    else:
        truth_keypoints = current_Points
        truth_boxes = current_boxes
    return truth_keypoints,truth_boxes
def InputReader(path):
    cap=cv2.VideoCapture("{}".format(path))  #视频流读取
    rate = cap.get(5)   # 帧速率
    FrameNumber = cap.get(7)  # 视频文件的帧数
    duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
    w = cap.get(3)   # 宽
    h = cap.get(4)  # 高
    return cap,rate,FrameNumber,duration,w,h


class track_output():

    def __init__(self,joint_number,pose_model,tracking_method,reid_model,Anormaly_dection,inputdir,projectname,save_vedio=True,save_json=True,show=True):
        self.model =  YOLO(pose_model)  
      
        if reid_model['status']:
            self.model2 = YOLO(reid_model['model_PATH'])
        # self.capture = capture
        # self.vedioname = os.path.basename(capture)
        self.joints = joint_number 
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

    def _inference(self,capture,write=False,show=False,json_write=False):
        mot_tracker = trackconfig.mutitracker(max_age=10, 
                min_hits=10,method=self.tracking_method,joints=self.joints) #create instance of the SORT tracker   
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
            out = cv2.VideoWriter(os.path.join(self.resultdir,self.tracking_method+'_'+vedioname+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), int(rate), (int(w), int(h)))
        while cap.isOpened():
 
            rec,img = cap.read()
            frameidx+=1
            if rec == False:  
                break
            # img = cv2.convertScaleAbs(img, alpha=0.8, beta=30)
            result = self.model(img, imgsz=640, conf=0.5,verbose=False)[0]
          
            if result.keypoints.conf != None:
                current_Points = result.keypoints.data.cpu().numpy() #获取全部点
                current_boxes = result.boxes.data[:,:-1].cpu().numpy()
                if len(current_Points) == 2 and flaga==False:  #进行初始化
                    history_Points = current_Points  
                    history_boxes = current_boxes
                    flaga = True
                if flaga == False:
                    continue
                if len(trackers)!=0 and flaga == True:
                    history_Points,history_boxes = Anomaly_detection(history_Points,current_Points,history_boxes,current_boxes) 
                # print(history_Points)
                combined_arr = np.stack(history_Points, axis=0)
                tp = combined_arr.reshape((len(history_boxes),-1,3))[:,:,:-1].reshape((len(history_boxes),-1))
                # print(tp)
                trackers,matchID = mot_tracker.update(tp) 
              

                # ---------- json numpy存储 ----------------------  不产生 npy 因为会产生很多个轨迹的id
                if json_write:
                    if len(matchID) !=0:
                        temppoints = np.copy(history_Points)
                        temppoints[:,:,0] =temppoints[:,:,0]/w
                        temppoints[:,:,1] =temppoints[:,:,1]/h
                        
                        ID = [x[1] for x in matchID]
                 
                        jsondict['keypoints'][frameidx] = {}
                        jsondict['boxes'][frameidx] ={}
                        for i,tracker in enumerate(trackers):
                            jsondict['keypoints'][frameidx][tracker[-1]] =  history_Points[ID[i]].tolist()
                            jsondict['boxes'][frameidx][tracker[-1]] = history_boxes[ID[i]].tolist()
                # ---------- -------- ----------------------

            custom.visiualize_tracker(img,result,trackers)
            cv2.putText(img, str(frameidx), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,200,255), 2)

            if write:
                out.write(img)  #视频写入           
            if show:
                cv2.namedWindow("enhanced",0)
                cv2.resizeWindow("enhanced", 1400, 600)
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
        


if __name__ == "__main__":
    PATH = 'D:/user/SZ/Project/DLC/vedio/' #'./predict/data/vedio/'
   
    a = track_output(5,'models/best_hand.pt','FP-sort',{'status': True,'model_PATH':'models/gender.pt'},{'status': True, 'fixnum': 2},PATH,'test1')
    a.muti_inference()





        
        


