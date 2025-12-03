import os
import shutil
import random

from tqdm import tqdm

# Dataset_root = 'Fish'
# original = 'original'
# image_path = 'images/'
# label_path = 'labelme_jsons/'
# format = '0'
# test_frac = 0.15

def exam_result(Dataset_root,original,format):   # 0 mistake  1 yolo  2 labelme   3 x-anylabel  4 coco
    path = os.path.join('dataset',Dataset_root, original)
    if format == 1:
        LISTTXT = [i for i in os.listdir(path) if i.split('.')[1] == 'txt']
        LISTimg = [i for i in os.listdir(path) if i.split('.')[1] == 'jpg' or i.split('.')[1] == 'png']
        if len(LISTTXT) == len(LISTimg):
            temp = 0
            for i in LISTimg:
                if i.split('.')[0]+'.txt' in LISTTXT:
                    temp+=1
            if temp == len(LISTimg):
                return True # txt
            else: 
                print('Please check if all images in the folder have corresponding txt files, or if there is an error in the input format pattern.')
                return False
        else:
            print('Please check if all images in the folder have corresponding txt files, or if there is an error in the input format pattern.')
            return False
    else:
        return False


def divide_data(Dataset_root,original,format,test_frac):
    if exam_result(Dataset_root,original,format):
        if format == 1:
            try:
                os.mkdir(os.path.join('dataset',Dataset_root, 'train'))
                os.mkdir(os.path.join('dataset',Dataset_root, 'val'))
            except:
                print('已经创建文件夹！！')
            originpath = os.path.join('dataset',Dataset_root, original)
            label_paths = [i for i in os.listdir(originpath) if i.split('.')[1] == 'txt']
            img_paths = [i for i in os.listdir(originpath) if i.split('.')[1] == 'jpg' or i.split('.')[1] == 'png']

            random.seed(123) 
            val_number = int(len(img_paths) * test_frac) 
            train_images = img_paths[val_number:]         
            val_images = img_paths[:val_number]           
            train_labels = label_paths[val_number:]        
            val_labels = label_paths[:val_number]          

            for i in train_images:
                shutil.move(os.path.join(originpath,i), os.path.join('dataset',Dataset_root, 'train'))
            for i in val_images:
                shutil.move(os.path.join(originpath,i), os.path.join('dataset',Dataset_root, 'val'))
            for i in train_labels:
                shutil.move(os.path.join(originpath,i), os.path.join('dataset',Dataset_root, 'train'))
            for i in val_labels:
                shutil.move(os.path.join(originpath,i), os.path.join('dataset',Dataset_root, 'val'))
            os.remove(originpath)
    
    

