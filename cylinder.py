import cv2
import numpy as np
class cylinder:
    def __init__(self,f=705) -> None:
        self.f = f

    def project(self,img):
        
        img = img[0]
        out_img = np.zeros_like(img,dtype=np.uint8)
        h,w,_ = img.shape
        img = img.flatten()
        s = self.f
        x_origin = np.floor(w / 2)
        y_origin = np.floor(h / 2)
        x_arange = np.arange(w)
        y_arange = np.arange(h)
        x_prime, y_prime = np.meshgrid(x_arange, y_arange)
        x_prime = x_prime - x_origin
        y_prime = y_prime - y_origin
        x = s * np.tan(x_prime / s)
        y = np.sqrt(x*x + s*s) / s * y_prime
        x += x_origin
        y += y_origin

        idx = np.ones([h, w])
        floor_x = np.floor(x).astype('int32')
        idx[floor_x < 0] = 0; idx[floor_x > w-1] = 0
        floor_x[floor_x < 0] = 0; floor_x[floor_x > w-1] = w-1

        ceil_x = np.ceil(x).astype('int32')
        idx[ceil_x < 0] = 0; idx[ceil_x > w-1] = 0
        ceil_x[ceil_x < 0] = 0; ceil_x[ceil_x > w-1] = w-1

        floor_y = np.floor(y).astype('int32')
        idx[floor_y < 0] = 0; idx[floor_y > h-1] = 0
        floor_y[floor_y < 0] = 0; floor_y[floor_y > h-1] = h-1

        ceil_y = np.ceil(y).astype('int32')
        idx[ceil_y < 0] = 0; idx[ceil_y > h-1] = 0
        ceil_y[ceil_y < 0] = 0; ceil_y[ceil_y > h-1] = h-1

        xt = ceil_x - x
        yt = ceil_y - y
        for c in range(3):
            left_up = img[c :: 3][floor_y*w + floor_x]
            right_up = img[c :: 3][floor_y*w + ceil_x]
            left_down = img[c :: 3][ceil_y*w + floor_x]
            right_down = img[c :: 3][ceil_y*w + ceil_x]
            t1 = left_up*xt + right_up*(1-xt)
            t2 = left_down*xt + right_down*(1-xt)

            out_img[:,:,c] = t1*yt + t2*(1-yt)

        out_img[idx == 0] = [0, 0, 0]
        
    
        return out_img
    
if __name__ == "__main__":

    img = cv2.imread('./image/0.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cyl = cylinder()
    test = cyl.project(img)
    cv2.imshow("test",test)
    cv2.waitKey()