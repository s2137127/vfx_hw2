from itertools import starmap
import numpy as np
from utils import *
from cylinder import *
import cv2
import os
from argparse import ArgumentParser
parse = ArgumentParser('High Dynamic Range Imaging')
parse.add_argument('--img_dir',default='./grail',type=str,help='directory for input images')
parse.add_argument('--use_cylinder',default=True,type=bool)
parse.add_argument('--feature_detection_method',default='sift',type=str,choices=['sift','harris'])
parse.add_argument('--feature_descriptor_method',default='sift',type=str,choices=['sift','neighbor'])
parse.add_argument('--save_matches_img',default=None,type=int,help='which matches to show')
parse.add_argument('--save_output',default=True,type=bool)
parse.add_argument('--end2end_alignment',default=True,type=bool)
parse.add_argument('--save_keypoint_img',default=None,type=int,help='which keypoint image to show')
args = vars(parse.parse_args())
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    path = args['img_dir']
    img_name_list = sorted([i for i in os.listdir(path) if i.endswith((".jpg",".png"))])
    img_list = [cv2.imread(os.path.join(path,i)) for i in img_name_list]
    
    f_matching = f_matching()
    det = f_detection()
    blend = blending()
    des = f_descriptor()
    if args['use_cylinder']:
        cyl = cylinder()
        f = open(os.path.join(path,'focal.txt'), 'r')
        focal = [float(line[:-1]) for line in f.readlines()]
        img_list = pool.starmap(cyl.project,[[i,f] for i,f in zip(img_list,focal)])
   
        
    img_list = [[i] for i in img_list]
    
    R = pool.starmap(det.get_R,img_list)
    print("get keypoints...")
    if args['feature_detection_method'] == 'sift':
        keypoint = pool.starmap(det.get_keypoint_sift,img_list) 
    else:
        keypoint = pool.starmap(det.get_keypoint_harris,[[R[i]] for i in range(len(R))])
    # print(type(keypoint[0]))
    keypoint = pool.starmap(det.get_keypoint_sift,img_list)  
    # print(type(keypoint[0]))
    if args['save_keypoint_img'] is not None:
        img_tmp  = img_list[args['save_keypoint_img']][0].copy()

        for i,j in keypoint[0]:
            cv2.circle(img_tmp,(j,i),2,(255,0,0))

        cv2.imwrite("kp_%s_%d.png" %(args['feature_detection_method'],args['save_keypoint_img']),img_tmp)
    
    print('get descriptor...')
    if args['feature_descriptor_method'] == 'sift':
         feature = pool.starmap(des.get_feature_sift,[(img[0],kp) for img,kp in zip(img_list,keypoint)])
    else:
        feature = pool.starmap(des.get_feature,[(img,kp) for [img],kp in zip(img_list,keypoint)])

   
    
    # print(len(feature))
    task = [[feature[i],feature[i+1]] for i in range(len(feature)-1)]
    matches = pool.starmap(f_matching.get_match,task)
    if args['save_matches_img'] is not None:
        idx = args['save_matches_img']
        draw_match_point(img_list[idx][0], keypoint[idx], img_list[idx+1][0], keypoint[idx+1], matches[idx],args['feature_detection_method'])
    img_match = img_matching()
    # draw_match_point(img_list[0][0], keypoint[0], img_list[1][0], keypoint[1], matches)
    # print([len(i) for i in matches])
    shift = pool.starmap(img_match.Ransac,[(keypoint[i],keypoint[i+1],matches[i]) for i in range(len(matches))])
    
    # print(shift)
    # print(shift[0])
    # print(img_list[0][1].shape)
    print("blending...")
    img = blend.stitch(shift[0],img_list[0][0],img_list[1][0],pool)
    for i in range(1,len(shift)):
        img = blend.stitch(shift[i],img,img_list[i+1][0],pool)
    
    # cv2.imshow("img_b",img)
    # cv2.waitKey()
    if args['end2end_alignment']:
        img = end2end(img,shift,[keypoint[0],keypoint[-1]],[feature[0],feature[-1]],f_matching,img_match)
        # cv2.imshow("img_blend",img)
        # cv2.waitKey()
    img = blend.crop(img)
    if args['save_output']:
        cv2.imwrite('./panaromas.png',img)
    # cv2.imshow("img_blend",img_t)
    # cv2.waitKey()
        # draw_match_point(img_list[0][0], keypoint[0], img_list[1][0], keypoint[1], matches)