import cv2
import numpy as np

#Q1 チャネル入れ替え
'''
img = cv2.imread('imori.jpg')
blue = img[:, :, 0].copy()
green = img[:, :, 1].copy()
red = img[:, :, 2].copy()

img[:, :, 0] = red
img[:, :, 1] = green
img[:, :, 2] = blue

cv2.imwrite('imori_r.jpg', img)
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#Q2 グレースケール化
'''
img = cv2.imread('imori.jpg')
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r =img[:, :, 2].copy()

y = 0.2126*r + 0.7152*g + 0.0722*b
y = y.astype(np.uint8)

cv2.imwrite('out.jpg', y)
cv2.imshow('result', y)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#03 二値化
'''
img = cv2.imread('imori.jpg')
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

y = 0.2126*r + 0.7152*g + 0.0722*b
y = y.astype(np.uint8)
th=128
y[y<th] = 0
y[y>=th] = 255

cv2.imshow('result', y)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#04 大津の二値化
'''
img = cv2.imread('imori.jpg').astype(np.float)

H,W,C = img.shape

#Grayscale
out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
out = out.astype(np.uint8)

#Determine threshold of Otsu's binarization
max_sigma = 0
max_t = 0

for _t in range(1, 255):
    v0 = out[np.where(out < _t)]
    m0 = np.mean(v0) if len(v0) > 0 else 0.
    w0 = len(v0) / (H * W)
    v1 = out[np.where(out >= _t)]
    m1 = np.mean(v1) if len(v1) > 0 else 0.
    w1 = len(v1) / (H * W)
    sigma = w0 * w1 * ((m0 - m1) ** 2)
    if sigma > max_sigma:
        max_sigma = sigma
        max_t = _t

# Binarization
print("threshold >>", max_t)
th = max_t
out[out < th] = 0
out[out >= th] = 255

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#05 HSV変換

