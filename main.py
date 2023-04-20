from itertools import starmap
import numpy as np
from utils import *
from cylinder import *
import cv2
import os
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    path = './parrington'
    img_name_list = sorted([i for i in os.listdir(path) if i.endswith((".jpg",".png"))])
    img_list = [cv2.imread(os.path.join(path,i)) for i in img_name_list]
    # tasks = [[i] for i in img_list]
    
    cyl = cylinder()
    det = f_detection()
    blend = blending()
    des = f_descriptor()

    # img_list = pool.starmap(cyl.project,tasks)
    while len(img_list ) >1:
        print("1")
        
        img_list = [[i] for i in img_list]
        
        R = pool.starmap(det.get_R,img_list)
        # print((R[0]))
        keypoint = pool.starmap(det.get_keypoint_harris,[[R[i]] for i in range(len(R))])
        # print(type(keypoint[0]))
        # keypoint = pool.starmap(det.get_keypoint_sift,img_list)  
        # print(type(keypoint[0]))   
        # img_tmp  = img_list[1][0].copy()
        # for i,j in keypoint[1]:
        #      cv2.circle(img_tmp,(j,i),2,(255,0,0))

        # cv2.imshow("img",img_tmp)
        # cv2.waitKey(0)
        # keypoint,cv2_kp = det.get_keypoint(R[0])
        # print("1")
        
        # feature = pool.starmap(des.get_feature,[(img,kp) for [img],kp in zip(img_list,keypoint)])

        feature = pool.starmap(des.get_feature_sift,[(img[0],kp) for img,kp in zip(img_list,keypoint)])
        matching = f_matching()
        print(len(feature))
        task = [[feature[2*i],feature[2*i+1]] for i in range(int(np.floor(len(feature)/2)))]
        matches = pool.starmap(matching.get_match,task)
        img_match = img_matching()
        # draw_match_point(img_list[0][0], keypoint[0], img_list[1][0], keypoint[1], matches)
        print([len(i) for i in matches])
        shift = pool.starmap(img_match.Ransac,[(keypoint[2*i],keypoint[2*i+1],matches[i]) for i in range(len(matches))])
        
        # print(shift)
        # print(shift[0])
        # print(img_list[0][1].shape)
        
        img_list = [blend.stitch(shift[i],img_list[2*i][0],img_list[2*i+1][0],pool) for i in range(len(shift))]
    img_t = blend.crop(img_list[0])
        # print(img_t.shape)
        # cv2.resize(img_t,dsize = [512,512])
    cv2.imshow("img_blend",img_t)
    cv2.waitKey()
        # draw_match_point(img_list[0][0], keypoint[0], img_list[1][0], keypoint[1], matches)