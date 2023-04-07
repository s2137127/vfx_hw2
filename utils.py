import numpy as np
import cv2
from scipy.spatial.distance import cdist
class f_detection:
    def __init__(self,gray_img):
        self.img = cv2.GaussianBlur(gray_img,(3,3),0)
        self.Ix =   cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        self.Iy =   cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        self.Ixx = self.Ix*self.Ix
        self.Iyy = self.Iy*self.Iy
        self.Ixy = self.Ix*self.Iy
        self.Sxx = np.zeros((self.img.shape[0]-2,self.img.shape[1]-2))
        self.Syy = np.zeros((self.img.shape[0]-2,self.img.shape[1]-2))
        self.Sxy = np.zeros((self.img.shape[0]-2,self.img.shape[1]-2))
        self.R = np.zeros_like(self.Ix)
        self.keypoint = []
    def sum_I(self):
        for i in range(1,self.img.shape[0]-2):
            for j in range(1,self.img.shape[1]-2):
                self.Sxx[i , j ] =  np.sum(self.Ixx[i-1:i+2,j-1:j+2])
                self.Syy[i , j ] = np.sum(self.Iyy[i - 1:i + 2, j - 1:j + 2])
                self.Sxy[i , j ] = np.sum(self.Ixy[i - 1:i + 2, j - 1:j + 2])

    def get_R(self,k=0.05):
        for i in range( self.Sxx.shape[0]):
            for j in range( self.Sxx.shape[1]):
                det = self.Sxx[i,j] * self.Syy[i,j] - self.Sxy[i,j]**2
                trace = self.Sxx[i,j] + self.Syy[i,j]
                self.R[i+1,j+1] = det-k*trace**2
        cv2.normalize(self.R, self.R, 0, 1, cv2.NORM_MINMAX)

    def get_keypoint(self,threshold=0.3,point_num=100):
        self.sum_I()
        self.get_R()
        point = np.array([[self.R[i,j],i,j]  for i in range(self.R.shape[0])
            for j in range(self.R.shape[1]) if  abs(self.R[i,j]) > threshold  ])

        # point = np.array(point)
        # distance = np.zeros((point.shape[0],point.shape[0]))
        print("//")
        distance = cdist(point[:,1:],point[:,1:])
        # print(distance[0])
        idx = np.argmax(point[:,0])
        # print(distance[tmp,:])
        print("//")
        self.keypoint.append(point[idx,1:])
        idx_arr = []
        for _ in range(point_num-1):
            tmp = np.argsort(distance[idx,:])
            for i in reversed(tmp):
                if i not in idx_arr:
                    idx_arr.append(i)
                    idx = i
                    self.keypoint.append(point[i,1:])
                    
                    break
        

        
        #             cv2.circle(self.img,(j,i),2,(255,0,0))
        for i,j in self.keypoint:
            i,j = int(i),int(j)
            cv2.circle(self.img,(i,j),2,(255,0,0))
        cv2.imshow("img",self.img)
        cv2.waitKey(0)

# class f_matching:
#
#
#
# class img_matching:
#
#
# class blending:
if __name__ == "__main__":
    img = cv2.imread('./prtn00.jpg',cv2.IMREAD_GRAYSCALE)
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = f_detection(img)
    det.get_keypoint()
