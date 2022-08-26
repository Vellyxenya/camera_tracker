import cv2
import numpy as np

# Create an image with text on it
# img = np.zeros((100, 400), dtype='uint8')
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'TheAILearner', (5, 70), font, 2, (255), 5, cv2.LINE_AA)

img = cv2.imread('img.jpg', 0)

img1 = img.copy()

# Structuring Element
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

dilated = cv2.erode(img, kernel, iterations=4)


# Create an empty output image to hold values
# thin = np.zeros(img.shape, dtype='uint8')
#
# # Loop until erosion leads to an empty set
# while cv2.countNonZero(img1) != 0:
#     # Erosion
#     erode = cv2.erode(img1, kernel)
#     # Opening on eroded image
#     opening = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
#     # Subtract these two
#     subset = erode - opening
#     # Union of all previous sets
#     thin = cv2.bitwise_or(subset, thin)
#     # Set the eroded image for next iteration
#     img1 = erode.copy()

cv2.imshow('dilated', dilated)
cv2.waitKey(0)
