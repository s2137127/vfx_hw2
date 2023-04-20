import numpy as np
import cv2
from scipy.spatial.distance import cdist
import multiprocessing as mp
class f_detection:
    def __init__(self):
       pass
    

    def get_R(self,img,k=0.04):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img ,(3,3),3)
        # self.Ix =   cv2.Sobel(self.blur_img, cv2.CV_64F, 0, 0, ksize=3)
        # self.Iy =   cv2.Sobel(self.blur_img, cv2.CV_64F, 0, 1, ksize=3)
        Iy,Ix =  np.gradient(blur_img)
        Ixx = Ix*Ix
        Iyy = Iy*Iy
        Ixy = Ix*Iy
        Sxx = cv2.GaussianBlur(Ixx, (3,3), 3)
        Syy = cv2.GaussianBlur(Iyy, (3,3), 3)
        Sxy = cv2.GaussianBlur(Ixy, (3,3), 3)
        # R = np.zeros_like(Ix)
        det =  Sxx * Syy -  Sxy**2
        trace = Sxx + Syy
        R = det-k*trace**2
        return R
        # cv2.normalize(self.R, self.R, 0, 1, cv2.NORM_MINMAX)

    # def get_keypoint(self,threshold=0.5,r=8):
    #     self.get_R()
    #     point = np.array([[self.R[i,j],i,j]  for i in range(5,self.R.shape[0]-5)
    #         for j in range(5,self.R.shape[1]-5) if  abs(self.R[i,j]) > threshold  ])
    #     for _,i,j in point:
    #         cv2.circle(self.img,(int(j),int(i)),2,(255,0,0))
    #     cv2.imshow("img",self.img)
    #     cv2.waitKey(0)
    #     print("origin point",len(point))
    #     distance = cdist(point[:,1:],point[:,1:])
    #     while len(point) != 0:
    #         idx = np.argmax(point[:,0])
    #         remain = [j for j in range(len(point)) if distance[idx,j]>r]
    #         self.keypoint.append(cv2.KeyPoint(point[idx,1],point[idx,2],1))
    #         point = point[remain]

    #     print("keypoints:",len(self.keypoint))
    #     # for i,j in self.keypoint:
    #     #     cv2.circle(self.img,(j,i),2,(255,0,0))
    #     # cv2.imshow("img",self.img)
    #     # cv2.waitKey(0)
    #     return self.keypoint
    def get_keypoint_harris(self,R,threshold=0.05,r=2):
        # cv2_keypoint=[]
        point = R
        point[np.where(abs(R) < np.max(R)*threshold)] = 0
        keypoint = point
        kp_list = []
        print('number of point',len(np.where(point != 0)[0]))
        for i in range(1,point.shape[0]-2):
            for j in range(1,point.shape[1]-2):
                if np.argmax(np.abs(point[i-1:i+2,j-1:j+2]).flatten()) != 4:
                    keypoint[i,j] = 0
        tmp = np.where(keypoint != 0)
        print('number of rest point',len(tmp[0]))
        # print(tmp[0])
        for i,j in zip(tmp[0],tmp[1]):
            # print(i,j)
            if i>1 and j>1 and i<R.shape[0] and j <R.shape[1]:
            #    cv2_keypoint.append(cv2.KeyPoint(int(i),int(j),1))
                kp_list.append([int(i),int(j)])

        #         cv2.circle(self.img,(j,i),2,(255,0,0))

        # cv2.imshow("img",self.img)
        # cv2.waitKey(0)
        return np.array(kp_list)#,cv2_keypoint
    
    def get_keypoint_sift(self,image,threshold=3.0,sigma=2**(1/4),num_octaves=2 ,num_DoG_images_per_octave=4 ):
        print(image.shape)
        image = cv2.cvtColor(image[0],cv2.COLOR_BGR2GRAY)
        global image_
        num_guassian_images_per_octave = num_DoG_images_per_octave + 1
        gaussian_images = []
        for i in range(num_octaves):
            gaussian_images.append(image)
            for j in range(num_guassian_images_per_octave - 1):
                image_ = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=sigma ** (j + 1))
                gaussian_images.append(image_)
               
            image = cv2.resize(image_, (int(image_.shape[1] / 2), int(image_.shape[0] / 2)),
                               interpolation=cv2.INTER_NEAREST)
        
        dog_images = [cv2.subtract(gaussian_images[i * 5 + j + 1], gaussian_images[i * 5 + j])
                      for i in range(num_octaves)
                      for j in range(num_DoG_images_per_octave)]

        # for i in range(num_octaves):
        #     for j in range(num_DoG_images_per_octave):
        #         a = abs(dog_images[i * 4 + j])
        #         m = np.max(a)
                # pic = (a * 255) / m
                # cv2.imwrite("./img%d_%d.png" % (i, j), pic.astype(np.int32))
        # print(dog_images[:2])
       
        keypoints = []
        for i in range(num_octaves):
            for j in range(1, num_DoG_images_per_octave - 1):
                b = 0
                for x in range(1, dog_images[i * 4 + j].shape[0] - 1):
                    for y in range(1, dog_images[i * 4 + j].shape[1] - 1):
                        neighbor = [dog_images[i * 4 + j][x - 1:x + 2, y - 1:y + 2],
                                    dog_images[i * 4 + j + 1][x - 1:x + 2, y - 1:y + 2],
                                    dog_images[i * 4 + j - 1][x - 1:x + 2, y - 1:y + 2]]
                        if np.max(neighbor) <= neighbor[0][1, 1] or np.min(neighbor) >= neighbor[0][1, 1]:
                            if abs(neighbor[0][1, 1]) >= threshold:
                                keypoints.append([x * (2 ** i), y * (2 ** i)])
                                # print(neighbor)
                                b += 1
        
        keypoints = np.unique(keypoints, axis=0)
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
     
class f_descriptor:
    def __init__(self) -> None:
        pass  
    def get_feature(self,img,keypoint,k_size =3):
        f=[]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3,3), 3)
        img_tmp = np.zeros((img.shape[0]+k_size-1,img.shape[1]+k_size-1))
        edge = int(np.floor(k_size/2))
        img_tmp[edge:-edge,edge:-edge] = img
        
        for p in keypoint:
            
            f.append(img_tmp[int(p[0]-np.floor(k_size/2)+edge):int(p[0]+np.ceil(k_size/2)+edge),
                                  int(p[1]-np.floor(k_size/2)+edge):int(p[1]+(np.ceil(k_size/2)+edge))].flatten() )
        # f = np.array([self.img[int(p.pt[0]-np.floor(k_size/2)):int(p.pt[0]+np.ceil(k_size/2)),
        #                           int(p.pt[1]-np.floor(k_size/2)):int(p.pt[1]+(np.ceil(k_size/2)))].flatten() 
        #                             for p in self.keypoint])
        return np.array(f)
    def get_feature_sift(self, img, keypoint, bins=8, k_size=9):
        # print("get_feature.........................")
        # get orientation

        _, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img, (3, 3), 3)
        Iy, Ix = np.gradient(blur_img)
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        M = (Ixx + Iyy) ** (1 / 2)
        degree = np.arctan(Iy / (Ix + 1e-8)) * (180 / np.pi)
        degree[Ix < 0] += 180
        degree = (degree + 360) % 360
        # print(degree)

        bin_size = 360. / bins  # 45degree
        theta_bins = (degree + (bin_size / 2)) // int(bin_size) % bins  # divide to 8 bins

        histo = np.zeros((bins,) + Ix.shape)
        for b in range(bins):
            histo[b][theta_bins == b] = 1
            histo[b] *= M
            histo[b] = cv2.GaussianBlur(histo[b], (k_size, k_size), 0)

        ori = np.argmax(histo, axis=0)

        # return ori, histo, degree, theta_bins, M

        # get_sub_vector
        bins, h, w = histo.shape

        def get_sub_vector(fy, fx, oy, ox, ori):
            sum = []
            for b in range(bins):
                sum.append(np.sum(ori[b][fy:fy + oy, fx:fx + ox]))

            sum_n1 = [x / (np.sum(sum) + 1e-8) for x in sum]
            sum_clip = [x if x < 0.2 else 0.2 for x in sum_n1]
            sum_n2 = [x / (np.sum(sum_clip) + 1e-8) for x in sum_clip]

            return sum_n2

        def get_vector(fpy, fpx, degree):
            # +angle in cv2 is counter-clockwise.
            # +y is down in image coordinates.
            M = cv2.getRotationMatrix2D((12, 12), degree[fpy, fpx], 1)
            if fpy - 12 < 0 or fpx - 12 < 0: return np.zeros((1,128))
            ori_rotated = [cv2.warpAffine(t[fpy - 12:fpy + 12, fpx - 12:fpx + 12], M, (24, 24)) for t in histo]

            vector = []
            suboffsets = [4, 8, 12, 16]
            for fy in suboffsets:
                for fx in suboffsets:
                    vector.append(get_sub_vector(fy, fx, 4, 4, ori_rotated))
            # print("vector:", vector)
            # print()
            # print(type(vector))
            return np.array(vector)

        descriptor = []
        # descriptors_left = []
        # descriptors_right = []
        for y, x in zip(keypoint[:, 0], keypoint[:, 1]):
            vectors = np.array(get_vector(y, x, degree)).flatten()
            if vectors.shape == (16,):
                print(vectors)
            descriptor.append(vectors)
            # if np.sum(vectors) > 0:
            #     if x <= width / 2:
            #         descriptors_left.append({'y': y, 'x': x, 'vector': vectors})
            #
            #     else:
            #         descriptors_right.append({'y': y, 'x': x, 'vector': vectors})
        # print('descriptors: (left: %d, right: %d)' % (len(descriptors_left), len(descriptors_right)))

        # descriptor = np.array(descriptor)
        # print(descriptor)

        # print([len(descriptor[i])for i in range(len(descriptor))])

        return np.array(descriptor)
 
class f_matching:
    def __init__(self) :
        pass
    def get_match(self,f1,f2,th_ratio=0.85):
        matches = []
        dist = cdist(f1,f2)
        for i in range(dist.shape[0]):
            idx = np.argsort(dist[i])
            if dist[i,idx[0]]<th_ratio*dist[i,idx[1]]:
            # print(dist[i,idx])
                matches.append((i,idx[0],dist[i,idx[0]]))
                # cv2_matches.append(cv2.DMatch(i,idx[0],dist[i,idx[0]]))
        print("matches:",len(matches))
        return matches#,cv2_matches
        


class img_matching:
    def __init__(self) -> None:
        pass
    def Ransac(self,kp1,kp2,matches,n=2,th=150):
        print(len(kp1))
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        matches = np.array(matches)[:,:2].astype(int)
        # print(matches[0,0])
        p=0.5
        P=0.999
        k = np.ceil(np.log(1-P)/np.log(1-p**n)).astype(int)
        dist,inlinear=[],[]
        for t in range(k):
            
            idx = np.random.randint(low=0,high=matches.shape[0],size=n)
            id1,id2 = matches[idx,0] ,matches[idx,1]
            p1,p2 =np.array([kp1[matches[i,0]] for i in range(matches.shape[0]) if i not in idx]),\
                    np.array([kp2[matches[i,1] ] for i in range(matches.shape[0]) if i not in idx])
            d = np.array(kp2[id2]-kp1[id1])
            d = np.array([np.sum(d[:,0]/n,dtype=int),np.sum(d[:,1]/n,dtype=int)])
            # print(d)
           
            err = np.subtract(np.add(p1,d),p2)
            err = np.sqrt(np.square(err[:,0])+np.square(err[:,1]))
            # print(err)
            inlinear.append(np.where(err<th)[0].shape[0])
            dist.append(d)
        idx = np.argmax(inlinear)
        # print(dist[idx])
        return dist[idx]
 
class blending:
    def __init__(self) -> None:
        pass

    def stitch(self,shift,img1,img2,pool):
          
        padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
        ]
        shifted_img1 = np.lib.pad(img1, padding, 'constant', constant_values=0)

        # cut out unnecessary region
        split = img2.shape[1]+abs(shift[1])
        splited = shifted_img1[:, split:] if shift[1] > 0 else shifted_img1[:, :-split]
        shifted_img1 = shifted_img1[:, :split] if shift[1] > 0 else shifted_img1[:, -split:]

        h1, w1, _ = shifted_img1.shape
        h2, w2, _ = img2.shape
        
        inv_shift = [h1-h2, w1-w2]
        inv_padding = [
            (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
            (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
            (0, 0)
        ]
        shifted_img2 = np.lib.pad(img2, np.abs(inv_padding), 'constant', constant_values=0)

        direction = 'left' if shift[1] > 0 else 'right'
        seam_x = shifted_img1.shape[1]//2
        tasks = [(shifted_img1[y], shifted_img2[y], seam_x, 2, direction) for y in range(h1)]
        shifted_img1 = pool.starmap(self.alpha_blend, tasks)
        shifted_img1 = np.asarray(shifted_img1)
        shifted_img1 = np.concatenate((shifted_img1, splited) if shift[1] > 0 else (splited, shifted_img1), axis=1)

        return shifted_img1
    def alpha_blend(self,row1, row2, seam_x, window, direction='left'):
        if direction == 'right':
            row1, row2 = row2, row1

        new_row = np.zeros(shape=row1.shape, dtype=np.uint8)

        for x in range(len(row1)):
            color1 = row1[x]
            color2 = row2[x]
            if x < seam_x-window:
                new_row[x] = color2
            elif x > seam_x+window:
                new_row[x] = color1
            else:
                ratio = (x-seam_x+window)/(window*2)
                new_row[x] = (1-ratio)*color2 + ratio*color1

        return new_row
    def crop(self,img):
        # cv2.imshow('te',img)
        # cv2.waitKey()
        print(img.shape)
        _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
        upper, lower = [-1, -1]

        black_pixel_num_threshold = img.shape[1]//5

        for y in range(thresh.shape[0]):
            if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
                upper = y
                break
            
        for y in range(thresh.shape[0]-1, 0, -1):
            if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
                lower = y
                break

        return img[upper:lower, :]
def draw_match_point(img1, keypoints1, img2, keypoints2, matches):
    # print(matches[0])
    # keypoints1 = keypoints1.astype(float)
    # keypoints2 = keypoints2.astype(float)
    # keypoints1 = [cv2.KeyPoint(i,j,1) for i,j in keypoints1]
    # keypoints2 = [cv2.KeyPoint(i,j,1) for i,j in keypoints2]
    # matches = [cv2.DMatch(i,j,k) for i,j,k in matches[0]]
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    # cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Good Matches', img_matches)
    # cv2.waitKey()
    assert img1.shape == img2.shape , "resize the image and coordinate"
    w = img1.shape[1]
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    img_matches[:,:w] = img1
    img_matches[:,w:] = img2
    
    for i,j,_ in matches[0]:
        color=np.random.randint(0,256,size=(1,3)).tolist()
        # print(color)
        cv2.circle(img_matches,keypoints1[i,::-1],radius=3,color=[255,0,0])
        cv2.circle(img_matches,(keypoints2[j,1]+w,keypoints2[j,0]),radius=3,color=[0,255,0])
        cv2.line(img_matches,keypoints1[i,::-1],(keypoints2[j,1]+w,keypoints2[j,0]),color=color[0],thickness=2)
    cv2.imshow('matches',img_matches)
    cv2.waitKey()



# if __name__ == "__main__":
    # img1 = cv2.imread('./prtn00.jpg')
    # img2 = cv2.imread('./prtn01.jpg')
    # det1 = f_detection(img1)
    # det2 = f_detection(img2)
    # keypoints1,cv2_kp1 = det1.get_keypoint()
    # keypoints2,cv2_kp2 = det2.get_keypoint()
    # feature1 = det1.get_feature()
    # feature2 = det2.get_feature()
    # matching = f_matching(feature1,feature2)
    # matches,cv2_matches =matching.get_match()
    # img_match = img_matching(keypoints1,keypoints2,matches)
    # shift = img_match.Ransac()
    # blend = blending()
    # print('cpu:',mp.cpu_count())
    # pool = mp.Pool(mp.cpu_count())
    # img_t = blend.stitch(shift,img1,img2,pool)
    # img_t = blend.crop(img_t)
    # cv2.imshow("imgg",img_t)
    # cv2.waitKey()
    # draw_match_point(img1, cv2_kp1, img2, cv2_kp2, cv2_matches)
