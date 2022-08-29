import cv2
import sys
import decoding
from decoding import decode
from datetime import datetime
import pickle


def do_nothing(*args):
    pass


def flip_binarizing(v):
    global do_binarizing
    do_binarizing = bool(v)


def update_dilations(v):
    global dilations
    dilations = v


def update_char_dist(v):
    # global new_character_distance
    decoding.new_character_distance = v


# Hyper parameters
dilations = 4
# new_character_distance = 60

# Initialize video capture
cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
if not ret:
    sys.exit('Could not set autofocus!')
cap.set(cv2.CAP_PROP_FOCUS, 150.0)

do_binarizing = False
init_loop = True

# Create the sliders
cv2.namedWindow('video', cv2.WINDOW_GUI_NORMAL)
cv2.createTrackbar('x_min', 'video', 121, 100, do_nothing)
cv2.createTrackbar('x_max', 'video', 169, 100, do_nothing)
cv2.createTrackbar('y_min', 'video', 256, 100, do_nothing)
cv2.createTrackbar('y_max', 'video', 97, 100, do_nothing)
cv2.createTrackbar('focus', 'video', 250, 250, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v))
cv2.createTrackbar('dilations', 'video', dilations, 7, update_dilations)
cv2.createTrackbar('character_width', 'video', decoding.new_character_distance, 80, update_char_dist)
cv2.createTrackbar('binarize', 'video', 1, 1, flip_binarizing)

# Initialize arrays to store measurements
times = []
A_list = []
Ah_list = []

start = datetime.now()

# Main loop
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

    # Invert black/white
    inverted = cv2.bitwise_not(gray)

    # Resize the image to have sth that is well-suited for parameter-tweaking
    height, width = inverted.shape
    scale_factor = 1200.0 / width
    inverted = cv2.resize(inverted, (1200, int(scale_factor * height)))

    # Decode the image
    decoded, inverted_with_bb = decode(inverted, dilations=dilations)
    decoded = decoded.split('\n')
    print(decoded)

    # Initialize times
    now = datetime.now()
    dt = now - start
    times.append(dt.total_seconds())
    A_list.append(None)
    Ah_list.append(None)

    # Add decoded characteres to list
    if len(decoded) == 4:  # If correctly formatted
        A = decoded[0]
        Ah = decoded[2]
        if len(A) > 0 and A[-1] == 'A':
            try:
                A = float(A[:-1])
                A_list[-1] = A
            except ValueError:
                print(f'A={A} is not a float')
        if Ah.endswith('Ah'):
            try:
                Ah = float(Ah[:-2])
                Ah_list[-1] = Ah
            except ValueError:
                print(f'Ah={Ah} is not a float')

    # Show the captured image on screen
    cv2.imshow('video', inverted_with_bb)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free resources
cap.release()
cv2.destroyAllWindows()

# Write the gathered data to disk
with open('times.pkl', 'wb') as f:
    pickle.dump(times, f)
with open('a_list.pkl', 'wb') as f:
    pickle.dump(A_list, f)
with open('ah_list.pkl', 'wb') as f:
    pickle.dump(Ah_list, f)
