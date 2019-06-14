# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""

import xml.etree.ElementTree as ET
import numpy as np
from scipy.misc import imsave
import cv2
import os
import openslide
from PIL import Image


def findExtension(directory, extension='.xml'):
    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files


def fillImage(image, coordinates, color=255):
    cv2.fillPoly(image, coordinates, color=color)
    return image


def readXML(filename):
    tree = ET.parse(filename)

    root = tree.getroot()
    regions = root[0][1].findall('Region')

    pixel_spacing = float(root.get('MicronsPerPixel'))

    labels = []
    coords = []
    length = []
    area = []

    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3

        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x, y]]

        coords += [coord]

    return coords, labels, length, area, pixel_spacing


def saveImage(image_size, coordinates, labels, sample=4):
    # red is 'benign', green is 'in situ' and blue is 'invasive'
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    img = np.zeros(image_size, dtype=np.uint8)

    for c, l in zip(coordinates, labels):
        img1 = fillImage(img, [np.int32(np.stack(c))], color=colors[l])
        img2 = img1[::sample, ::sample, :]
    return img2

def getGT(xmlpath, scan, sample, level):
    dims = scan.dimensions
    img_size = (dims[1], dims[0], 3)

    coords, labels, length, area, pixel_spacing = readXML(xmlpath)
    gt = saveImage(img_size, coords, labels, sample=sample)

    gt = Image.fromarray(gt).convert('RGB').resize(scan.level_dimensions[level])
    gt = np.asarray(gt)
    gt = np.concatenate((np.zeros((gt.shape[0], gt.shape[1], 1)), gt), axis=-1)
    gt = np.argmax(gt, axis=-1)

    return gt
