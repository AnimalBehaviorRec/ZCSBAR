import numpy as np
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


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

    return angle

def distance_batch(pp_test,pp_gt,method,K=0):   #(x1-x2)**2+(y1-y2)**2+....(xn-xn)**2
    #sort   bytetrack   FP-sort    P-sort    FP-delta-sort 
    temp =False
    try:
      if method == 'P-sort':
        result = np.linalg.norm(pp_test[:,:2][:, np.newaxis, :] - pp_gt[:,:2], axis=2)
      elif method == 'FP-sort':
        result = np.linalg.norm(pp_test[:, np.newaxis, :] - pp_gt, axis=2)
      elif method == 'FP-delta-sort':
        angle_test = np.array([angle_between_vectors(pp_test[i][2:4],pp_test[i][0:2],[0,0],[0,1]) for i in range(len(pp_test))])
        angle_gt =   np.array([angle_between_vectors(pp_gt[i][2:4],pp_gt[i][0:2],[0,0],[0,1]) for i in range(len(pp_gt))])
        Points_distance = np.linalg.norm(pp_test[:, np.newaxis, :] - pp_gt, axis=2)    #  修改[:,:2]
        angle =  abs( (angle_test[:, np.newaxis] - angle_gt)/180 )
        result = Points_distance+K*angle
      else:
        temp=True
      # print(Points_distance,angle)
 
    except:
      print(pp_test,pp_gt)
      return []
    if temp:
      print('check whether you set the right tracking method or not!')
      raise NameError
    return result  #欧式距离  Points_distance+K*angle
    
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, points,joints):
        """
        Initialises a tracker using initial points.
        """
        # Define constant velocity model
        # Each point has (x, y), so 5 points have 10 dimensions plus their velocities (dx, dy) = 20 dimensions
        self.num = joints
        self.vector_dimension = self.num*2
        self.kf = KalmanFilter(dim_x=self.vector_dimension*2, dim_z=self.vector_dimension)

        # State transition matrix
        self.kf.F = np.eye(self.vector_dimension*2)
        for i in range(self.vector_dimension):
            self.kf.F[i, i+self.vector_dimension] = 1

        # Measurement function
        self.kf.H = np.zeros((self.vector_dimension, self.vector_dimension*2))
        for i in range(self.vector_dimension):
            self.kf.H[i, i] = 1

        # Measurement uncertainty
        self.kf.R[self.num:, self.num:] *= 0.8

        self.kf.P *= 0.03
        self.kf.Q[self.vector_dimension:, self.vector_dimension:] *= 0.4
        # Initial state
        self.kf.x[:self.vector_dimension] = points.reshape(self.vector_dimension, 1)

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
        self.kf.update(points.reshape(self.vector_dimension, 1))

    def predict(self):
        """
        Advances the state vector and returns the predicted points estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:self.vector_dimension].reshape((self.num, 2)))
        return self.history[-1]
    def get_state(self):
      """
      Returns the current bounding box estimate.
      """
      return self.kf.x[:self.vector_dimension].reshape((1,self.vector_dimension))

def associate_points_to_trackers(points,trackers,method,K,distance_threshold = 40**2):   #将5个的平方数相加  (10,1)
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_points and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(points)), np.empty((0,3),dtype=int)
  
  #print('pointc:',points,'\n','oldtrackers:',trackers)
  
  # for i in range(5):
  distance_matrix = distance_batch(points, trackers[:,:-1],method,K=K)
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

class mutitracker(object):
  def __init__(self, method='',joints=5,max_age=1,K=0, min_hits=3, max_distance_threshold=30**2):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits      #  最少进行 update次数 ，有匹配
    self.max_distance_threshold = max_distance_threshold
    self.trackers = []
    self.frame_count = 0
    self.K = K
    self.joints = joints
    if method == '':
      self.method = 'FP-sort'
    self.method = method
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
    matched, unmatched_points, unmatched_trks = associate_points_to_trackers(points,trks,self.method,self.K,self.max_distance_threshold)  #1
    # print(unmatched_points,'\n')
    # update matched trackers with assigned detections  
    for m in matched:
      self.trackers[m[1]].update(points[m[0], :])
      
    # print(points,matched)

    # create and initialise new trackers for unmatched detections
    for i in unmatched_points:
        # print(points[i,:])
        trk = KalmanBoxTracker(points[i,:],self.joints)
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