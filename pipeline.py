import os
import cv2 as cv
import numpy as np


def extract_characters(input_image):
    """
    Given an input image, extract the contained characters from top-to-bottom, left-to-right
    :param input_image: input image as a numpy array. Must contain only one channel
    :return: sorted bounding boxes for each character, Nones for white spaces
    """
    # Extract contours
    threshold = 100
    canny_output = cv.Canny(input_image, threshold, threshold * 2)
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create bounding boxes
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    # Some contours are inside of others. Mark them as invalid
    invalid_indices = []
    for i in range(len(centers)):
        for j in range(len(centers)):
            if j == i:
                continue
            if radius[i] < radius[j] and cv.pointPolygonTest(contours_poly[j], centers[i], True) > 0:  # point is inside
                invalid_indices.append(i)

    # Only select the valid bounding boxes
    bounding_boxes = []
    for i in range(len(contours)):
        if i not in invalid_indices:
            bounding_boxes.append(boundRect[i])

    # Split the bounding boxes into a top and bottom row
    midline = 120  # y coordinate of center line between the top and the bottom rows
    row_1 = list(filter(lambda elem: elem[1] < midline, bounding_boxes))
    row_2 = list(filter(lambda elem: elem[1] >= midline, bounding_boxes))

    # Sort each row along the x axis
    row_1 = sorted(row_1, key=lambda elem: elem[0])
    row_2 = sorted(row_2, key=lambda elem: elem[0])

    # Merge the bounding boxes again
    bounding_boxes = np.array(row_1 + row_2)

    def compute_distance(bb1, bb2):
        """
        Distance between anchors of bounding boxes
        :param bb1: first bounding box
        :param bb2: second bounding box
        :return:
        """
        v1 = np.array([bb1[0], bb1[1]])
        v2 = np.array([bb2[0], bb2[1]])
        return np.linalg.norm(v1 - v2)

    # Initialize final bounding box list
    final_bounding_boxes = [bounding_boxes[0]]

    # Ignore bounding boxes that are close to the previous ones
    for bb_ in bounding_boxes[1:]:
        dist = compute_distance(final_bounding_boxes[-1], bb_)
        if dist > 200:  # If distance is large, we have a white space
            final_bounding_boxes.append(None)
        if dist > 20:  # If distance is not too small we have a new character
            final_bounding_boxes.append(bb_)

    return final_bounding_boxes


def template_matching(input_character):
    """
    Given a character, look for corresponding template id by performing template-matching
    :param input_character: the character to match
    :return: id of matching template
    """
    max_matching = 0
    arg_i = -1
    for i, template_ in enumerate(templates):
        res = cv.matchTemplate(input_character, template_, cv.TM_CCOEFF_NORMED).flatten()
        res = max(res)  # Select the best matching
        if res > max_matching:
            arg_i = i
            max_matching = res
    return arg_i


if __name__ == '__main__':
    # Read input image
    img = cv.imread('img.jpg', 0)

    # Preprocess the image
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # Remark: erode b/c the characters are black. Would dilate instead if they were white
    dilated = cv.erode(img, kernel, iterations=4)  # dilate the characters
    src_gray = cv.blur(dilated, (3, 3))

    # Read and resize the templates
    templates = []  # list of templates
    template_names = []  # list of corresponding characters
    for image in os.listdir('templates'):
        template = cv.imread(os.path.join('templates', image), 0)
        height, width = template.shape
        scale_factor = 120.0 / height
        template = cv.resize(template, (int(scale_factor * width), 120))
        templates.append(template)
        template_names.append(image.split('.')[0])

    # map template names to characters
    dico = {
        'a': 'A',
        'w': 'W',
        'v': 'V',
        'point': '.'
    }

    # Character extraction phase
    bbs = extract_characters(src_gray)

    # Matching phase
    decoded = ''
    for bb in bbs:
        if bb is None:  # If None we have a white space
            decoded += '\n'
            continue
        x, y, w, h = bb
        # Extract the character from image before preprocessing
        character = img[y:y+h, x:x+w]
        # Resize the image
        height, width = character.shape
        scale_factor = 120.0 / height
        character = cv.resize(character, (int(scale_factor * width), 120))
        # Add spaces on the left and the right to ensure the image is always wider than the template
        height, width = character.shape
        background = np.zeros((120, 240))
        background[:, 120 - width//2: 120 - width//2 + width] = character
        # Perform template matching
        template_id = template_matching(character)
        # Add the decoded character to the final string
        decoded += dico.get(template_names[template_id], template_names[template_id])

    print('Decoded:\n')
    print(decoded)



