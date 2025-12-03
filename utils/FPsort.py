import numpy as np
from filterpy.kalman import KalmanFilter
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
import glob
import time


np.random.seed(0)

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def distance_batch(pp_test,pp_gt):   #(x1-x2)**2+(y1-y2)**2+....(xn-xn)**2
    try:
      
      result = np.linalg.norm(pp_test[:, np.newaxis, :] - pp_gt, axis=2)
    except:
      print(pp_test,pp_gt)
      return []
    return result  #欧式距离
    


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, points):
        """
        Initialises a tracker using initial points.
        """
        # Define constant velocity model
        # Each point has (x, y), so 5 points have 10 dimensions plus their velocities (dx, dy) = 20 dimensions
        self.kf = KalmanFilter(dim_x=20, dim_z=10)

        # State transition matrix
        self.kf.F = np.eye(20)
        for i in range(10):
            self.kf.F[i, i+10] = 1

        # Measurement function
        self.kf.H = np.zeros((10, 20))
        for i in range(10):
            self.kf.H[i, i] = 1

        # Measurement uncertainty
        self.kf.R[5:, 5:] *= 0.8

        self.kf.P *= 0.03
        self.kf.Q[10:, 10:] *= 0.4
        # Initial state
        self.kf.x[:10] = points.reshape(10, 1)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, points):
        """
        Updates the state vector with observed points.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(points.reshape(10, 1))

    def predict(self):
        """
        Advances the state vector and returns the predicted points estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:10].reshape((5, 2)))
        return self.history[-1]
    def get_state(self):
      """
      Returns the current bounding box estimate.
      """
      return self.kf.x[:10].reshape((1,10))



def associate_points_to_trackers(points,trackers,distance_threshold = 40**2):   #将5个的平方数相加
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_points and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(points)), np.empty((0,3),dtype=int)
  
  #print('pointc:',points,'\n','oldtrackers:',trackers)
  
  # for i in range(5):
  distance_matrix = distance_batch(points, trackers[:,:-1])
  # print('distance:',distance_matrix)

  if min(distance_matrix.shape) > 0:
    # a = (distance_matrix < distance_threshold).astype(np.int32)
    # if a.sum(1).max() == 1 and a.sum(0).max() == 1:
    #     matched_indices = np.stack(np.where(a), axis=1)
    # else:
    matched_indices = linear_assignment(distance_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))
  
  # print('matched_indices:',matched_indices)

  unmatched_points = []
  for d, det in enumerate(points):
    if(d not in matched_indices[:,0]):
      unmatched_points.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(distance_matrix[m[0], m[1]] > distance_threshold):
      unmatched_points.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)
  # print(matches)
  return matches, np.array(unmatched_points), np.array(unmatched_trackers)


class FPSort(object):
  def __init__(self, max_age=1, min_hits=3, max_distance_threshold=30**2):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits      #  最少进行 update次数 ，有匹配
    self.max_distance_threshold = max_distance_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, points=np.empty((0, 2))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 11))
    to_del = []
    point = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape(1,10)[0]  #[[1,1],[1,1],[1,1],[1,1],[1,1]]
     
      trk[:] = np.append(pos,0)   # x,y,0  ---> x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,0
  
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_points, unmatched_trks = associate_points_to_trackers(points,trks, self.max_distance_threshold)  #1
    # print(unmatched_points,'\n')
    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(points[m[0], :])
      
    # print(points,matched)

    # create and initialise new trackers for unmatched detections
    for i in unmatched_points:
        # print(points[i,:])
        trk = KalmanBoxTracker(points[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):   #reverse 倒序    可用在数组更新与删除
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          point.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(point)>0):
      #print(points,point,matched,unmatched_points)
      return np.concatenate(point),matched
    return np.empty((0,3)),matched




if __name__ == '__main__':
  # all train
  display = False
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display   显示颜色   有帧数间隔
  if(display):
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join('fish', '*', 'det', 'det.txt')  # 'data\\*\\det\\det.txt'   # 有两个选择  det_anormal_dectect  det
  for seq_dets_fn in glob.glob(pattern):     # seq_dets_fn 'data\\240123\\det\\det.txt'

    mot_tracker = FPSort(max_age=20, 
                       min_hits=2,
                       max_distance_threshold=30**2) #create instance of the SORT tracker  # 替换跟踪器
    
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')    # seq_dets shape(4325,10)   # 核心代码1  此处修改检测器
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]  #'240123'      
    
    with open(os.path.join('fish',seq,'result', '%s.txt'%(seq+'-det-FPsort')),'w') as out_file:
      print("Processing %s."%(seq))   
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1  
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]   #
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]     pointa,b,c,d,e
        points = np.array(seq_dets[seq_dets[:, 0]==frame, 7:22]).reshape((len(dets),5,3))[:,:,:-1].reshape((len(dets),10))      # 7-10 10-13 13-16 16-19 19-21
        total_frames += 1  
        if(display):
          fn = os.path.join('fish', seq, 'images', '%07d.jpg'%((frame-1)*3))      # phase = train  seq = ADL-Rundle-6  ADL-Rundle-8
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')
        start_time = time.time()
        trackers,matchID = mot_tracker.update(points)     # 核心代码2
        
        cycle_time = time.time() - start_time
        total_time += cycle_time

        cost_matrix = distance_batch(trackers[:,:-1],points)   #将预测的点与原点进行匹配
        dic ={}
        if len(cost_matrix)!=0:
          MATCH = linear_assignment(cost_matrix)   
          for i in MATCH:
            dic[i[0]] = i[1]
      
        for i,tracker in enumerate(trackers):   # point从后往前 因为sort类包含了 reverse
          ID = int(tracker[10])                            #目标编号
          if len(trackers)!=0: 
            d = dets[dic[i]]                                   # 在dets中对应的顺序
          else:
            d = dets[i]
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,int(tracker[10]),d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)    
          # print('%d,%d,%.2f,%.2f,1,-1,-1,-1'%(frame,tracker[2],tracker[0],tracker[1]),file=out_file)      
          if(display):
            # d = d.astype(np.int32)
            tracker = tracker.astype(np.int32)
            # ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[ID%32,:]))
            ax1.scatter(tracker[0], tracker[1], color=colours[ID % 32, :], s=10)
            ax1.scatter(tracker[2], tracker[3], color=colours[ID % 32, :], s=10)
            ax1.scatter(tracker[4], tracker[5], color=colours[ID % 32, :], s=10)
            ax1.scatter(tracker[6], tracker[7], color=colours[ID % 32, :], s=10)
            ax1.scatter(tracker[8], tracker[9], color=colours[ID % 32, :], s=10)

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
