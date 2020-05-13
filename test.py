from cv2 import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(0)

# arr = []
# arr2 = [1,2,3,4]
# arr.extend(arr2)
# print(arr)


# back_ground = np.ones((100, 100), dtype=np.uint8)
# back_ground = cv.cvtColor(back_ground, cv.COLOR_GRAY2BGR)
# back_ground[:, :, :] = 0  # Black background
# cv.imshow('test',back_ground)
# cv.waitKey()
infos = []
infos.extend([(0.0, 0.0, 1.0, 1.0)])
print('infos:',infos)