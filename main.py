import time

import cv2
import sys
from pipeline import decode

# template_0 = cv2.imread('0.png', cv2.IMREAD_GRAYSCALE)
# element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# template_0_dilated = cv2.dilate(template_0, element, iterations=2)
# cv2.imshow('templ', template_0_dilated)
# cv2.waitKey(0)
# sys.exit(0)


def do_nothing(*args):
    pass


do_binarizing = False
def flip_binarizing(v):
    global do_binarizing
    do_binarizing = bool(v)


cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
# if not ret:
#     sys.exit('Could not set autofocus!')

cap.set(cv2.CAP_PROP_FOCUS, 150.0)

cv2.namedWindow('video', cv2.WINDOW_GUI_NORMAL)
cv2.createTrackbar('x_min', 'video', 123, 100, do_nothing)
cv2.createTrackbar('x_max', 'video', 169, 100, do_nothing)
cv2.createTrackbar('y_min', 'video', 179, 100, do_nothing)
cv2.createTrackbar('y_max', 'video', 17, 100, do_nothing)
cv2.createTrackbar('focus', 'video', 250, 250, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v))
cv2.createTrackbar('binarize', 'video', 1, 1, flip_binarizing)

A_list = []
Ah_list = []

init_loop = True
while True:
    ret, frame = cap.read()
    if init_loop:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        cv2.setTrackbarMax('x_min', 'video', frame_width)
        cv2.setTrackbarMax('x_max', 'video', frame_width)
        cv2.setTrackbarMax('y_min', 'video', frame_height)
        cv2.setTrackbarMax('y_max', 'video', frame_height)
        initLoop = False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray[cv2.getTrackbarPos('y_min', 'video'):int(frame_height / 2) + cv2.getTrackbarPos('y_max', 'video'),
                cv2.getTrackbarPos('x_min', 'video'):int(frame_width / 2) + cv2.getTrackbarPos('x_max', 'video')]

    if do_binarizing:
        ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Decode
    inverted = cv2.bitwise_not(gray)

    height, width = inverted.shape
    scale_factor = 1200.0 / width
    inverted = cv2.resize(inverted, (1200, int(scale_factor * height)))

    decoded = decode(inverted)

    decoded = decoded.split('\n')
    print(decoded)
    print('---------------')

    if len(decoded) == 4:  # Correctly formatted
        A = decoded[0]
        Ah = decoded[2]
        if A[-1] == 'A':
            try:
                A = float(A[:-1])
                A_list.append(A)
            except ValueError:
                print(f'A is not a float, it is: {A}')

        if Ah.endswith('Ah'):
            try:
                Ah = float(Ah[:-2])
                Ah_list.append(Ah)
            except ValueError:
                print(f'Ah is not a float, it is: {Ah}')

    print(len(Ah_list))
    if len(Ah_list) == 30:
        break

    cv2.imshow('video', inverted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(A_list)
print()
print(Ah_list)
