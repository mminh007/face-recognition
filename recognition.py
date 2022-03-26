import cv2
import argparse
import numpy as np
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
import insightface
import utils


assert insightface.__version__>='0.3'

parser = argparse.ArgumentParser(description='insightface app test')
# general
parser.add_argument('--ctx', default=1, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')
args = parser.parse_args(args = [])
app = FaceAnalysis()
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))


location = ['Billy','Donaire','Holly','Hugh', 'Jake', 'Jon', 'Jonathan', 'Joseph', 'Judge', 'Karen', 'Kelly', 'Lilith', 'Lily', 'Linh', 'Milly', 'Minh', 'Philips', 'Ryan', 'Scarlet', 'Scott', 'Thao', 'Tifa', 'Tim', 'Tom', 'Tuan', 'Winston']
add = 'D:/DS/Datasets'


data = utils.choose_imgs(location, add) 
print(data)

# Data embedding
database = utils.database_embedding_faces(data)


# Image test
img_path = 'D:/DS/Datasets/Face-Data/87711.jpg'
cv2.imread(img_path)
img = utils.image_embedding_faces(img_path)


dict_bbox = utils.get_most_similar_face(img, database)


# Draw bbox
draw = utils.draw_bbox_img(img_path, dict_bbox, (255,0,0), 2)