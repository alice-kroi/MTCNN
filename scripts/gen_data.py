import sys
import os
# 添加父目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from PIL import Image
import numpy as np
from models.modules.utils import offset_boxes
import random
from models.modules.iou import iou
#from MTCNN.tool import utils
import traceback
 
anno_src = r"E:\github\dataset\celeba\list_bbox_celeba.txt"                      #原来的样本数据（在生成样本时使用）
img_dir = r"E:\github\dataset\celeba\img_celeba"                    #源图片（用于生成新样本）
 
save_path = r"E:\github\MTCNN\datasets\iuput"                                  #生成样本的总的保存路径
 
float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]    
# 用于生成样本的随机数，用于生成样本的框的偏移量
def gen_data(save_path, anno_src, img_dir,face_size,stop_value):
    postive_image_dir = os.path.join(save_path, "positive")     #仅仅生成路径名
    negative_image_dir = os.path.join(save_path, "negative")
    part_image_dir = os.path.join(save_path, "part")


    for dir_path in [postive_image_dir, negative_image_dir, part_image_dir]:     #生成路径
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    postive_anno_filename = os.path.join(save_path, "positive.txt")
    negative_anno_filename = os.path.join(save_path, "negative.txt")
    part_anno_filename = os.path.join(save_path, "part.txt")

    postive_count = 0
    negative_count = 0
    part_count = 0
    postive_anno_file = open(postive_anno_filename, "w")
    negative_anno_file = open(negative_anno_filename, "w")
    part_anno_file = open(part_anno_filename, "w")
    for i, line in enumerate(open(anno_src)):      #遍历每一行
        if i < 2:
            continue
        strs = line.split()
        image_filename = strs[0].strip()          #置信度   #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        # print(image_filename)
        image_file = os.path.join(img_dir, image_filename)
        if not os.path.exists(image_file):
            print("{} not exist".format(image_file))
            continue
        # print(image_file)
        with Image.open(image_file) as img:
            img_w, img_h = img.size
            x1 = float(strs[1].strip())
            y1 = float(strs[2].strip())
            w = float(strs[3].strip())           #人脸框
            h = float(strs[4].strip())
            x2 = float(x1 + w)
            y2 = float(y1 + h)
            # print(x1,y1,x2,y2)
            if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0 or (w*h)/max(w*h)^2<=0.7:    #人脸框的大小不能小于40，否则不生成
                continue
            boxes = np.array([[x1, y1, x2, y2]])    #人脸框
            # print(boxes)
            offset = offset_boxes(boxes)       #人脸框的偏移量
            # print(offset)
            '''
            px1=float(strs[5].strip())
            py1=float(strs[6].strip())
            px2=float(strs[7].strip())
            py2=float(strs[8].strip())
            px3=float(strs[9].strip())
            py3=float(strs[10].strip())
            px4=float(strs[11].strip())
            py4=float(strs[12].strip())
            px5=float(strs[13].strip())
            py5=float(strs[14].strip())
            # print(px1,py1,px2,py2,px3,py3,px4,py4,px5,py5)
            landmarks=np.array([[px1,py1],[px2,py2],[px3,py3],[px4,py4],[px5,py5]])   #人脸框的五个关键点
            # print(landmarks)
            '''
            cx,cy=x1+w/2,y1+h/2#求出人脸框的中心点
            side_len=random.choice([w,h])
            single_img_pos=0
            single_img_neg=0
            single_img_part=0
            while True:
                if single_img_pos<3:
                    _side_len=side_len*(random.uniform(0.8,1.2))+1
                    _cx=_cx*random.uniform(0.8,1.2)+1
                    _cy=_cy*random.uniform(0.8,1.2)+1
                elif single_img_neg<3:
                    _side_len=side_len*(random.uniform(-1,3))+1
                    _cx=_cx*random.uniform(-1,3)+1
                    _cy=_cy*random.uniform(-1,3)+1
                elif single_img_part<3:
                    _side_len=side_len*(random.uniform(0,2))+1
                    _cx=_cx*random.uniform(0,2)+1
                    _cy=_cy*random.uniform(0,2)+1

                #计算生成样本的坐标，生成样本为正方形
                _x1=_cx-_side_len/2 # 偏移中心反算原坐标
                _y1=_cy-_side_len/2
                _x2=_cx+_side_len   #获得偏移后的x2y2
                _y2=_cy+_side_len
                # print(_x1,_y1,_x2,_y2)
                if _x1<0 or _y1<0 or _x2>img_w or _y2>img_h or _side_len<face_size:
                    continue
                offset_x1 = (x1 - _x1) / _side_len                      #得到四个偏移量
                offset_y1 = (y1 - _y1) / _side_len
                offset_x2 = (x2 - _x2) / _side_len
                offset_y2 = (y2 - _y2) / _side_len
 
                offset_px1 = 0#(px1 - x1_) / side_len     #offset偏移量
                offset_py1 = 0#(py1 - y1_) / side_len
                offset_px2 = 0#(px2 - x1_) / side_len
                offset_py2 = 0#(py2 - y1_) / side_len
                offset_px3 = 0#(px3 - x1_) / side_len
                offset_py3 = 0#(py3 - y1_) / side_len
                offset_px4 = 0#(px4 - x1_) / side_len
                offset_py4 = 0#(py4 - y1_) / side_len
                offset_px5 = 0#(px5 - x1_) / side_len
                offset_py5 = 0#(py5 - y1_) / side_len

                crop_box = [_x1, _y1, _x2, _y2]
                face_crop = img.crop(crop_box)       #图片裁剪
                face_resize = face_crop.resize((face_size, face_size)) 
                iou = iou(crop_box, np.array(boxes))[0]
                if iou > 0.65 :
                    postive_anno_file.write(
                        "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(
                            postive_count, offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5
                        )
                    )
                    postive_anno_file.flush()
                    face_resize.save(os.path.join(postive_image_dir, "{}.jpg".format(postive_count)))
                    postive_count += 1
                    single_img_pos+=1
                elif 0.7>iou > 0.3 :
                    part_anno_file.write(
                        "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(
                            part_count, offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5
                        )  
                    )
                    part_anno_file.flush()
                    face_resize.save(os.path.join(part_image_dir, "{}.jpg".format(part_count)))
                    part_count += 1
                    single_img_part+=1
                elif iou < 0.1 :
                    negative_anno_file.write(
                        "negative/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9}\n".format(
                            negative_count, offset_x1, offset_y1, offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5
                        ) 
                    )
                    negative_anno_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, "{}.jpg".format(negative_count)))
                    negative_count += 1
                    single_img_neg+=1
                count = postive_count+part_count+negative_count
                print(count)
                if count >= stop_value:
                    break