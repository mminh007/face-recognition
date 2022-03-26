
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import sys
import numpy as np
import os
from PIL import Image
from insightface.app import FaceAnalysis
import insightface


# assert insightface.__version__>='0.3'
parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=1, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args(args = [])
app = FaceAnalysis()
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

def choose_imgs(filename, address):  # chon 5 image lam database
    k_list = []
    for l in filename:
        path = address + l + '/'
        
        for k in os.listdir(path)[:5]:
            k_list.append(os.path.join(path, k))
            
    return(k_list)    


def database_embedding_faces(file):  #embedding feature extraction vector
  feature_faces = []
  list_names = []       
  for m in file:
    img = np.asarray(Image.open(m))
    faces = app.get(img)
    names = m.split('/')[-2]
    list_names.append(names)
    feature_faces.append(faces[0].embedding)              
  #images = np.array(images, dtype = np.float32)
  return list_names, feature_faces


# 1 ảnh có thể gồm nhiều faces
def image_embedding_faces(image_path):
  feature_faces = [] 
  bbox = []
  img = np.asarray(cv2.imread(image_path))
  faces = app.get(img)
  for i in range(len(faces)): # Số faces định vị được trong ảnh
    feature_faces.append(faces[i].embedding)
    bbox.append(faces[i].bbox)

  return bbox, feature_faces


def get_most_similar_face(img_input, database):
  names = database[0]
  features = database[1]
  
  bbox_input = img_input[0]
  feature_input = img_input[1]
  
  bbox = []
  names_bbox = []

  for i in range(len(feature_input)):
    distance = np.linalg.norm(np.array(features) - np.array(feature_input[i].reshape(1,-1)), axis = -1)  # compute Eculidean distance
    nearest = np.argmin(distance)   # -> list
    bbox.append(bbox_input[i])
    
    if nearest >= 0.6:
      names_bbox.append('Unknow')
      
    else:
      names_bbox.append(names[nearest[0]])

  return bbox, names_bbox




def draw_bbox_img(img_path, dict_bbox, color, thickness):
  names = dict_bbox[1]
  bboxs = dict_bbox[0]  # list bboxs
  img = cv2.imread(img_path)
  for name, bbox in zip(names, bboxs):
  # draw bbox
    x1, y1, x2, y2 = bbox  # Tọa độ giá trị int
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

  # draw a rectangle that contains label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    label_size, baseline = cv2.getTextSize(name, font, font_scale, 2)   # Tính kích thước label

    cv2.rectangle(img, (x1, int(y2 - label_size[1])), (int(x1 + label_size[0]), y2), (0, 146, 230), -1)  # vẽ khung label (khung màu cam))
    
    # draw label
    cv2.putText(img, name, (x1, y2), font, font_scale, color=(0, 0, 0)) 

  return cv2.imshow(img)

# abc
