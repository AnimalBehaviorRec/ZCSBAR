from ultralytics import YOLO
import os

current_path = os.path.abspath(os.path.dirname(__file__))
print(current_path)
def genarator_yaml(TrainYamlpath,Pramas):
    import yaml
    data = {
        'path': 'datasets',
        'train': os.path.abspath(Pramas['train']),
        'val': os.path.abspath(Pramas['val']),
        'kpt_shape':Pramas['kpt_shape'],
        'names': Pramas['names']
    }
    with open(os.path.join(TrainYamlpath,'train.yaml'), 'w') as file:
        yaml.dump(data, file,default_flow_style=False)
    
    if data['train']=='' or data['val']=='' or data['kpt_shape']=='' or type(data['kpt_shape'])!=list or type(data['names'])!=dict:
        print('please check training input value')
        return False
    else:
        return True

def train_model(Pramas,Dataset_root,pretrain=False):
    if pretrain == True:
        model = YOLO('models/'+Pramas['model'])
    else:
        model = YOLO(Pramas['model'])
    
    TrainYamlpath = os.path.join('dataset',Dataset_root)
    abspath = os.path.abspath(TrainYamlpath)
    if genarator_yaml(abspath,Pramas):
        model.train(data=os.path.join(abspath,'train.yaml'),epochs=Pramas['epochs'],imgsz=Pramas['imgsz'],batch=Pramas['batch'],cache=True,device=Pramas['device'],workers=0) #lr0=0.001, lrf=0.001c