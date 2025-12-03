import json
import os
from PIL import Image
from datetime import datetime
def yolo2coco_batch(image_folder, coco_folder,projectname):
    # 创建输出文件夹（如果不存在）
    os.makedirs(coco_folder, exist_ok=True)

    coco_data = {
        "images": [],
        "categories": [{
            "id": 1,
            "name": "fish",
            "supercategory": "fish",
            "color": "#7c7c7d",
            "metadata": {},
            "keypoint_colors": ["#bf5c4d", "#d99100", "#4d8068", "#0d2b80", "#9c73bf"],
            "keypoints": ['1', '2', '3','4','5'],
            "skeleton": [[0,1], [1,2], [2,3],[3,4]]
        }],
        "annotations": []
    }

    image_id = 0
    ann_id = 0

    for filename in os.listdir(image_folder):
        if filename.endswith(".txt"):
            yolo_filepath = os.path.join(image_folder, filename)
            image_filepath = os.path.join(image_folder, os.path.splitext(filename)[0] + ".png")
            #coco_filepath = os.path.join(coco_folder, os.path.splitext(filename)[0] + ".json")
            # Get image dimensions
            image = Image.open(image_filepath)
            image_width, image_height = image.size
            image_name = os.path.splitext(os.path.basename(image_filepath))[0]

            with open(yolo_filepath, 'r') as yolo_file:
                lines = yolo_file.readlines()

            yolo_annotations = []
            for line in lines:

                
                data = line.strip().split()

                bbox_relative = [float(data[i]) for i in range(1, 5)]
                keypoints_relative = [float(data[i]) for i in range(5, len(data))]

                bbox = [
                    ((bbox_relative[0] - bbox_relative[2] / 2) * image_width),
                    ((bbox_relative[1] - bbox_relative[3] / 2) * image_height),
                    (bbox_relative[2] * image_width),
                    (bbox_relative[3] * image_height)
                ]
                segmentation = []
                # #框的四个顶点坐标,必须填写，否则不显示框，只修改点可设置为空
                segmentation = [[bbox[0]+bbox[2],bbox[1],
                                bbox[0]+bbox[2],bbox[1]+bbox[3],
                                bbox[0],bbox[1]+bbox[3],
                                bbox[0],bbox[1]]]

                keypoints = []
                for i in range(len(keypoints_relative)):
                    if i % 3 == 0:
                        keypoints.append(round(keypoints_relative[i] * image_width))
                    elif i % 3 == 1:
                        keypoints.append(round(keypoints_relative[i] * image_height))
                    else:
                        if(keypoints_relative[i]>0.9):
                            keypoints.append(2)#标注且可见
                        else:
                            keypoints.append(1)#标注但遮挡

                annotation = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": "#343635",
                    "keypoints": keypoints,
                    "metadata": {},
                    "num_keypoints": len(keypoints) // 3
                }
                yolo_annotations.append(annotation)
                ann_id +=1

            coco_data["images"].append({
                "id": image_id,
                "dataset_id": 12,
                "path": image_filepath,
                "width": image_width,
                "height": image_height,
                "file_name": f"{image_name}.png"
            })

            coco_data["annotations"].extend(yolo_annotations)

            image_id += 1
            # ann_id += len(yolo_annotations)
    coco_filepath = os.path.join(coco_folder, projectname+'.json')
    with open(coco_filepath, 'w') as coco_file:
        json.dump(coco_data, coco_file, indent=4)

# Example Usage

# PATH1 = 'D:/user/SZ/dataset/FISH_COCO/imgs/'
# for imgfoder in os.listdir(PATH1):
#     print(PATH1+imgfoder)
#     yolo2coco_batch(image_folder=PATH1+imgfoder,
#                     coco_folder='annotations',projectname=imgfoder)

yolo2coco_batch(image_folder='guiyihua/test',
                    coco_folder='guiyihua/annotations',projectname='test')


