import os
import cv2 as cv
import numpy as np

import random as rng
rng.seed(12345)


def extract_characters(input_image, draw_on, verbose=False):
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
            if boundRect[i][2] < 10 or boundRect[i][3] < 10 or \
                    (radius[i] < radius[j] and cv.pointPolygonTest(contours_poly[j], centers[i], True) > 0):  # point is inside
                invalid_indices.append(i)

    # Only select the valid bounding boxes
    bounding_boxes = []
    for i in range(len(contours)):
        if i not in invalid_indices:
            bounding_boxes.append(boundRect[i])

    # Split the bounding boxes into a top and bottom row
    midline = input_image.shape[0] / 2  # y coordinate of center line between the top and the bottom rows
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

    def compute_distance_x(bb1, bb2):
        """
        X-distance between anchors of bounding boxes
        :param bb1: first bounding box
        :param bb2: second bounding box
        :return:
        """
        return abs(bb1[0] - bb2[0])

    if verbose:
        print(len(bounding_boxes))

    # Initialize final bounding box list
    if len(bounding_boxes) < 2:
        return [None]
    final_bounding_boxes = [bounding_boxes[0]]

    # Ignore bounding boxes that are close to the previous ones
    for bb_ in bounding_boxes[1:]:
        dist = compute_distance_x(final_bounding_boxes[-1], bb_)
        if dist > 200:  # If distance is large, we have a white space
            final_bounding_boxes.append(None)
        if dist > 60:  # If distance is not too small we have a new character
            final_bounding_boxes.append(bb_)

    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    for i in range(len(final_bounding_boxes)):
        if final_bounding_boxes[i] is None:
            continue
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.rectangle(draw_on, (int(final_bounding_boxes[i][0]), int(final_bounding_boxes[i][1])), \
                     (int(final_bounding_boxes[i][0] + final_bounding_boxes[i][2]),
                      int(final_bounding_boxes[i][1] + final_bounding_boxes[i][3])), color, 2)

    if verbose:
        cv.imshow('Contours', draw_on)
        cv.imshow('input', input_image)
        cv.waitKey(0)

    return final_bounding_boxes, draw_on


def template_matching(input_character, templates, names):
    """
    Given a character, look for corresponding template id using template-matching
    :param input_character: the character to match
    :return: id of matching template
    """
    max_matching = 10000
    arg_i = -1

    for i, template_ in enumerate(templates):
        name = names[i]
        res = cv.matchTemplate(input_character, template_, cv.TM_SQDIFF_NORMED).flatten()  # TODO Might need to change the matching algo b/c confuses 1 and 4
        res = min(res)  # Select the best matching
        if res < max_matching:
            arg_i = i
            max_matching = res
    return arg_i


def decode(input_image, dilations, verbose=False):
    """

    :param input_image:
    :param verbose:
    :return:
    """
    # Read input image
    img = input_image

    # Preprocess the image
    kernel_cross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    kernel_oblique_cross_3x3 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)
    # Remark: erode b/c the characters are black. Would dilate instead if they were white
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel_cross)
    dilated = cv.erode(img, kernel_oblique_cross_3x3, iterations=dilations)  # dilate the characters
    blurred = cv.blur(dilated, (3, 3))

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
    rgb_input_image = cv.cvtColor(dilated, cv.COLOR_GRAY2BGR)
    bbs, image_with_bbs = extract_characters(blurred, rgb_input_image, verbose=verbose)

    # Matching phase
    decoded = ''
    for bb in bbs:
        if bb is None:  # If None we have a white space
            decoded += '\n'
            continue
        x, y, w, h = bb
        # Extract the character from image before preprocessing
        character = img[y:y + h, x:x + w]
        # Resize the image
        height, width = character.shape
        ww = 600
        if height == 0 or width == 0 or width > ww:
            continue  # TODO but this should not happen..
        scale_factor = 120.0 / height

        character = cv.resize(character, (int(scale_factor * width), 120))

        # Add spaces on the left and the right to ensure the image is always wider than the template
        height, width = character.shape

        background = np.ones((120, ww)) * 255
        ww = 240
        if height == 0 or width == 0 or width > ww:
            continue  # TODO but this should not happen..

        background[:, ww//2 - width // 2: ww//2 - width // 2 + width] = character
        # Perform template matching
        template_id = template_matching(character, templates, template_names)
        # Add the decoded character to the final string
        decoded += dico.get(template_names[template_id], template_names[template_id])

    if verbose:
        print('Decoded:\n')
        print(decoded)

    return decoded, image_with_bbs


if __name__ == '__main__':
    image = cv.imread('new.png', 0)
    decode(image, 4, verbose=True)



