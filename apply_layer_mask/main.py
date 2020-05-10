import cv2
import sys
import numpy as np

_, crop_path, mask_path, out_path = sys.argv

crop_img = cv2.cvtColor(cv2.imread(crop_path), cv2.COLOR_BGR2BGRA)

mask_img = cv2.resize(
    cv2.imread(mask_path),
    np.shape(crop_img)[:2][::-1],
    interpolation = cv2.INTER_CUBIC)
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
temp = np.ones(np.shape(crop_img))
temp[:, :, -1] = mask_img
mask_img = np.divide(temp, [1, 1, 1, 255])

masked_img = np.multiply(crop_img, mask_img)

cv2.imwrite(out_path, masked_img)

