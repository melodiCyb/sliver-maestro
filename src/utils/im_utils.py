import cv2
import numpy as np
from skimage.morphology import skeletonize, area_opening
from skimage.util import invert
from svg.path import parse_path
import pandas as pd
from xml.dom import minidom
from utils import vrep
import time
from PIL import Image
import array
import matplotlib.pyplot as plt


def cut(img):
    # crop image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    new_img = img[y:y + h, x:x + w]
    return new_img


def transBg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, img.dtype)
    cv2.fillPoly(mask, roi, (255,) * img.shape[2], )
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def fourChannels(img):
    height, width, channels = img.shape
    if channels < 4:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return new_img
    return img


def thinning(input_path):
    src = cv2.imread(input_path, 0)
    _, src2 = cv2.threshold(src, 10, 255, cv2.THRESH_BINARY)
    image = invert(src2) / 255
    skeleton_lee = skeletonize(image, method='lee')
    # convert back black lines
    b_skeleton_lee = cv2.bitwise_not(skeleton_lee)
    return b_skeleton_lee


def remove_inside(image, connectivity=2):
    opened_im = area_opening(image, 64, connectivity=connectivity)
    return opened_im


def parse_svg(svg_file, csv_path):
    """
    Parses svg file into coordinates and saves as a csv file.
    svg_file: str. 
    csv_path: str.
    coordinates: ndarray. [x, y, is_stroke_end] 
    """

    x_coordinates = []
    y_coordinates = []
    z_coordinates = []
    doc = minidom.parse(svg_file)
    paths = doc.getElementsByTagName("path")

    for idx, e in enumerate(paths):
        next_path = parse_path(e.attributes["d"].value)
        x_coordinates.append(next_path[0].start.real)
        y_coordinates.append(next_path[0].start.imag)
        z_coordinates.append(0.0)
        for offset_idx in range(len(next_path)):
            x_coordinates.append(next_path[offset_idx].end.real)
            y_coordinates.append(next_path[offset_idx].end.imag)
            z_coordinates.append(0.0)
        z_coordinates[-1] = 1.0
    coordinates = np.column_stack((np.asarray(x_coordinates), np.asarray(y_coordinates), np.asarray(z_coordinates)))
    df = pd.DataFrame(coordinates, columns=['X(m)', 'Y(m)', 'Z(m)'])
    df.to_csv(csv_path)


def stream_vision_sensor(visionSensorName, clientID, pause=0.0001):
    # Get the handle of the vision sensor
    res1, visionSensorHandle = vrep.simxGetObjectHandle(clientID, visionSensorName, vrep.simx_opmode_oneshot_wait)
    # print visionSensorHandle
    # Get the image
    option = 1
    # res2,resolution,image = vrep.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, vrep.simx_opmode_streaming)
    # Allow the display to be refreshed
    plt.ion()
    # Initialiazation of the figure
    time.sleep(0.5)
    res, resolution, image = vrep.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, vrep.simx_opmode_buffer)
    im = Image.new("RGB", (resolution[0], resolution[1]), "white")
    # Give a title to the figure
    fig = plt.figure(1)
    fig.canvas.set_window_title(visionSensorName)
    # inverse the picture
    plotimg = plt.imshow(im, origin='lower')
    # Let some time to Vrep in order to let him send the first image, otherwise the loop will start with an empty image and will crash
    time.sleep(1)
    while (vrep.simxGetConnectionId(clientID) != -1):
        # Get the image of the vision sensor
        option = 1  # 0
        res, resolution, image = vrep.simxGetVisionSensorImage(clientID, visionSensorHandle, option,
                                                               vrep.simx_opmode_buffer)
        # Transform the image so it can be displayed using pyplot
        image_byte_array = array.array('b', image)
        im = Image.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
        # Update the image
        plotimg.set_data(im)
        # Refresh the display
        plt.draw()
        # The mandatory pause ! (or it'll not work)
        plt.pause(pause)
    print('End of Simulation')
