import time
import cv2
import numpy as np
img_path=r"E:\github\MTCNN\datasets\iuput\000001.jpg"
anno_path=r"E:\github\MTCNN\datasets\iuput\000001.txt"
for i,line in enumerate(open(anno_path)):
    if i<2:
        continue
    else:
        strs=line.split()
        x1=float(strs[0].strip())
        y1=float(strs[1].strip())
        x2=float(strs[2].strip())
        y2=float(strs[3].strip())
        img=cv2.imread(img_path)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        cv2.imshow("img",img)
        cv2.waitKey(0)
        time.sleep(3)
        