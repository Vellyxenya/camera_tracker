import cv2
import pytesseract
import sys

img = cv2.imread('img.jpg')
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Gueddach\AppData\Local\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(img, lang='fxs2',
                                   config='--psm 6 --oem 3 -c tessedit_char_whitelist=/0123456789ApVW.')
print(text)
# print(text.replace('\n', '').replace('\f', ''))

# C:\Users\Gueddach\AppData\Local\Tesseract-OCR

# def decode_fourcc(v):
#     v = int(v)
#     return ''.join([chr((v >> 8 * i) & 0xFF) for i in range(4)])
#
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# color = (0, 255, 0)
#
# cap = cv2.VideoCapture(0)
# focus_val = cap.get(cv2.CAP_PROP_FOCUS)
# print('focus value:', focus_val)
# ret = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
# if not ret:
#     sys.exit('Could not set autofocus!')
#
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# focus = int(min(cap.get(cv2.CAP_PROP_FOCUS) * 100, 2**31-1))  # ceil focus to C_LONG as Python3 int can go to +inf
#
# cv2.namedWindow('video')
# cv2.createTrackbar('FPS', 'video', fps, 30, lambda v: cap.set(cv2.CAP_PROP_FPS, v))
# cv2.createTrackbar('Focus', 'video', focus, 100, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v / 100))
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
#     cv2.putText(frame, 'Mode: {}'.format(fourcc), (15, 40), font, 1.0, color)
#     cv2.putText(frame, 'FPS: {}'.format(fps), (15, 80), font, 1.0, color)
#     cv2.imshow('video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()