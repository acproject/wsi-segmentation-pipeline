import numpy as np

import itertools
import numpy as np


def angle_with_start(coord, start):
    vec = coord - start
    return np.angle(np.complex(vec[0], vec[1]))


def sort_clockwise(points):
    # convert into a coordinate system
    # (1, 1, 1, 2) -> (1, 1), (1, 2)
    coords = [np.array([points[i], points[i+4]]) for i in range(len(points) // 2)]

    # find the point closest to the origin,
    # this becomes our starting point
    coords = sorted(coords, key=lambda coord: np.linalg.norm(coord))
    start = coords[0]
    rest = coords[1:]

    # sort the remaining coordinates by angle
    # with reverse=True because we want to sort by clockwise angle
    rest = sorted(rest, key=lambda coord: angle_with_start(coord, start), reverse=True)

    # our first coordinate should be our starting point
    rest.insert(0, start)
    # convert into the proper coordinate format
    # (1, 1), (1, 2) -> (1, 1, 1, 2)
    return list(itertools.chain.from_iterable(zip(*rest)))

def evenly_spaced_points_on_a_contour(points, num_pts):

    '''
    from functools import reduce
    import operator
    import math
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    points = sorted(points, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    '''

    points = np.asarray(points)

    x, y = points[:, 0], points[:, 1]

    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd ** 2 + yd ** 2)
    u = np.cumsum(dist)
    u = np.hstack([[0], u])

    t = np.linspace(0, u.max(), num_pts)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)

    es_points = np.vstack((xn, yn)).swapaxes(0, 1)

    return es_points


def _evenly_spaced_points_on_a_contour(points, num_pts):

    N = num_pts
    pX, pY = points[:, 0], points[:, 1]

    # equally spaced in arclength
    N = np.transpose(np.linspace(0, 1, N))

    # how many points will be uniformly interpolated?
    nt = N.size

    # number of points on the curve
    n = pX.size
    pxy = np.array((pX, pY)).T
    p1 = pxy[0, :]
    pend = pxy[-1, :]
    last_segment = np.linalg.norm(np.subtract(p1, pend))
    epsilon = 10 * np.finfo(float).eps

    # IF the two end points are not close enough lets close the curve
    if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
        pxy = np.vstack((pxy, p1))
        nt = nt + 1
    else:
        print('Contour already closed')

    pt = np.zeros((nt, 2))

    # Compute the chordal arclength of each segment.
    chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    tbins = np.digitize(N, cumarc)  # bin index in which each N is in

    # catch any problems at the ends
    tbins[np.where(tbins <= 0 | (N <= 0))] = 1
    tbins[np.where(tbins >= n | (N >= 1))] = n - 1

    s = np.divide((N - cumarc[tbins]), chordlen[tbins - 1])
    pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

    return pt


import numpy as np
import scipy.interpolate as sp
import math
import csv


def diffCOL(matrix):
    newMAT = []
    newROW = []
    for i in range(len(matrix) - 1):
        for j in range(len(matrix[i])):
            diff = matrix[i + 1][j] - matrix[i][j]
            newROW.append(diff)
        newMAT.append(newROW)
        newROW = []
    # Stack the matrix to get xyz in columns
    newMAT = np.vstack(newMAT)
    return newMAT


def squareELEM(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j] * matrix[i][j]
    return matrix


def sumROW(matrix):
    newMAT = []
    for j in range(len(matrix)):
        rowSUM = 0
        for k in range(len(matrix[j])):
            rowSUM = rowSUM + matrix[j][k]
        newMAT.append(rowSUM)
    return newMAT


def sqrtELEM(matrix):
    for i in range(len(matrix)):
        matrix[i] = math.sqrt(matrix[i])
    return matrix


def sumELEM(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum = sum + matrix[i]
    return sum


def diffMAT(matrix, denom):
    newMAT = []
    for i in range(len(matrix)):
        newMAT.append(matrix[i] / denom)
    return newMAT


def cumsumMAT(matrix):
    first = 0
    newmat = []
    newmat.append(first)
    # newmat.append(matrix)
    for i in range(len(matrix)):
        newmat.append(matrix[i])
    cum = 0
    for i in range(len(newmat)):
        cum = cum + newmat[i]
        newmat[i] = cum
    return newmat


def divMAT(A, B):
    newMAT = []
    for i in range(len(A)):
        newMAT.append(A[i] / B[i])
    return newMAT


def minusVector(t, cumarc):
    newMAT = []
    for i in range(len(t)):
        newMAT.append(t[i] - cumarc[i])
    return newMAT


def replaceIndex(A, B):
    newMAT = []
    for i in range(len(B)):
        index = B[i]
        newMAT.append(A[index])
    return newMAT


def matSUB(first, second):
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j] - second[i][j])
        # newMAT.append(newCOL)
    return newMAT


def matADD(first, second):
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j] + second[i][j])
        # newMAT.append(newCOL)
    return newMAT


def matMULTI(first, second):
    """
    Take in two matrix
    multiply each element against the other at the same index
    return a new matrix
    """
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j] * second[i][j])
        # newMAT.append(newCOL)
    return newMAT


def matDIV(first, second):
    """
    Take in two matrix
    multiply each element against the other at the same index
    return a new matrix
    """
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j] / second[i][j])
        # newMAT.append(newCOL)
    return newMAT


def vecDIV(first, second):
    """
    Take in two arrays
    multiply each element against the other at the same index
    return a new array
    """
    newMAT = []
    for i in range(len(first)):
        newMAT.append(first[i] / second[i])
    return newMAT


def replaceROW(matrix, replacer, adder):
    newMAT = []
    if adder != 0:
        for i in range(len(replacer)):
            newMAT.append(matrix[replacer[i] + adder])
    else:
        for i in range(len(replacer)):
            newMAT.append(matrix[replacer[i]])
    return np.vstack(newMAT)


def interparc(points, t):

    # Should check to make sure t is a single integer greater than 1
    t = np.linspace(0, 1, t)

    nt = len(t)

    px = points[:, 0]
    py = points[:, 1]
    n = len(px)

    pxy = [px, py]
    # pxy = np.transpose(pxy)
    ndim = 2

    method = 'linear'
    ndim = len(pxy)

    pt = np.zeros((nt, ndim))
    # Check for rounding errors here
    # Transpose the matrix to align with matlab codes method
    pxy = np.transpose(pxy)
    chordlen = sqrtELEM(sumROW(squareELEM(diffCOL(pxy))))
    chordlen = diffMAT(chordlen, sumELEM(chordlen))
    cumarc = cumsumMAT(chordlen)
    if method == 'linear':
        inter = np.histogram(bins=t, a=cumarc)
        tbins = inter[1]
        hist = inter[0]
        tbinset = []
        index = 0
        tbinset.append(index)

        for i in range(len(hist)):
            if hist[i] > 0:
                index = index + hist[i]
                tbinset.append(index)
            else:
                tbinset.append(index)

        for i in range(len(tbinset)):
            if tbinset[i] <= 0 or t[i] <= 0:
                tbinset[i] = 1
            elif tbinset[i] >= n or t[i] >= 1:
                tbinset[i] = n - 1
        # Take off one value to match the way matlab does indexing
        for i in range(len(tbinset)):
            tbinset[i] = tbinset[i] - 1

        s = divMAT(minusVector(t, replaceIndex(cumarc, tbinset)), replaceIndex(chordlen, tbinset))

        # Breakup the parts of pt
        repmat = np.transpose(np.reshape(np.vstack(np.tile(s, (1, ndim))[0]), (ndim, -1)))
        sub = np.reshape(np.vstack(matSUB(replaceROW(pxy, tbinset, 1), replaceROW(pxy, tbinset, 0))), (-1, ndim))
        multi = np.reshape(np.vstack(matMULTI(sub, repmat)), (-1, ndim))
        pt = np.reshape(np.vstack(matADD(replaceROW(pxy, tbinset, 0), multi)), (-1, ndim))
        return pt