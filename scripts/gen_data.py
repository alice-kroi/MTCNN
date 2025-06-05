import sys
import os
# 添加父目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from PIL import Image
import numpy as np
from models.modules.iou import iou
#from MTCNN.tool import utils
import traceback
 
anno_src = r"E:\github\dataset\celeba\list_bbox_celeba.txt"                      #原来的样本数据（在生成样本时使用）
img_dir = r"E:\github\dataset\celeba\img_celeba"                    #源图片（用于生成新样本）
 
save_path = r"E:\github\MTCNN\datasets\iuput"                                  #生成样本的总的保存路径
 
float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]       

