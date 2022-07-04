import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
UPFACE_POINTS = list(range(0, 48))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
MOUTH_POINTS = list(range(48, 68))

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(PREDICTOR_PATH)

#Compute Eyes
def eye_size(eye):
    eyeWidth = dist.euclidean(eye[0], eye[3])
    hull = ConvexHull(eye)
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)

    eyeCenter = eyeCenter.astype(int)

    return int(eyeWidth), eyeCenter


def place_eye(frame, eyeCenter, eyeSize):
    eyeSize = int(eyeSize * 1.5)

    x1 = int(eyeCenter[0, 0] - (eyeSize / 2))
    x2 = int(eyeCenter[0, 0] + (eyeSize / 2))
    y1 = int(eyeCenter[0, 1] - (eyeSize / 2))
    y2 = int(eyeCenter[0, 1] + (eyeSize / 2))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

        # re-calculate the size to avoid clipping
    eyeOverlayWidth = x2 - x1
    eyeOverlayHeight = y2 - y1

    # calculate the masks for the overlay
    eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask_eye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_eye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Nose
def nose_size(nose):
    noseWidth = dist.euclidean(nose[0], nose[8])
    hull = ConvexHull(nose)
    noseCenter = np.mean(nose[hull.vertices, :], axis=0)

    noseCenter = noseCenter.astype(int)

    return int(noseWidth), noseCenter

def place_nose(frame, noseCenter, noseSize):
    noseSize = int(noseSize * 1.5)

    x1 = int(noseCenter[0, 0] - (noseSize / 2))
    x2 = int(noseCenter[0, 0] + (noseSize / 2))
    y1 = int(noseCenter[0, 1] - (noseSize / 2) * 0.4)
    y2 = int(noseCenter[0, 1] + (noseSize / 2) * 0.4)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    noseOverlayWidth = x2 - x1
    noseOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    noseOverlay = cv2.resize(imgNose, (noseOverlayWidth, noseOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_nose, (noseOverlayWidth, noseOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_nose, (noseOverlayWidth, noseOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(noseOverlay, noseOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Hat
def hat_size(hat):
    hatWidth = dist.euclidean(hat[0], hat[9])
    hatCenter = hat[10]

    return int(hatWidth), hatCenter

def place_hat(frame, hatCenter, hatSize):
    hatSize = int(hatSize * 1.5)

    x1 = int(hatCenter[0, 0] - (hatSize / 2))
    x2 = int(hatCenter[0, 0] + (hatSize / 2))
    y1 = int(hatCenter[0, 1]*0.8 - (200*hatSize / 201)*0.9)
    y2 = int(hatCenter[0, 1]*0.8 + (hatSize / 201)*0.9)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    hatOverlayWidth = x2 - x1
    hatOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    hatOverlay = cv2.resize(imgHat, (hatOverlayWidth, hatOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_hat, (hatOverlayWidth, hatOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_hat, (hatOverlayWidth, hatOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(hatOverlay, hatOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Mostache
def moustache_size(moustache):
    moustacheWidth = dist.euclidean(moustache[0], moustache[6])
    hull = ConvexHull(moustache)
    moustacheCenter = np.mean(moustache[hull.vertices, :], axis=0)

    moustacheCenter = moustacheCenter.astype(int)

    return int(moustacheWidth), moustacheCenter

def place_moustache(frame, moustacheCenter, moustacheSize):
    moustacheSize = int(moustacheSize * 1.5)

    x1 = int(moustacheCenter[0, 0] - (moustacheSize / 2))
    x2 = int(moustacheCenter[0, 0] + (moustacheSize / 2))
    y1 = int(moustacheCenter[0, 1] - (moustacheSize / 3))
    y2 = int(moustacheCenter[0, 1] + (moustacheSize / 2))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    moustacheOverlayWidth = x2 - x1
    moustacheOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    moustacheOverlay = cv2.resize(imgMoustache, (moustacheOverlayWidth, moustacheOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_moustache, (moustacheOverlayWidth, moustacheOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_moustache, (moustacheOverlayWidth, moustacheOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(moustacheOverlay, moustacheOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Close Mouth
def mouth_size(mouth):
    mouthWidth = dist.euclidean(mouth[0], mouth[6])
    hull = ConvexHull(mouth)
    mouthCenter = np.mean(mouth[hull.vertices, :], axis=0)

    mouthCenter = mouthCenter.astype(int)

    distMouth = mouth[18, 1] - mouth[14, 1]

    return int(mouthWidth), mouthCenter, distMouth

def place_mouth(frame, mouthCenter, mouthSize):
    mouthSize = int(mouthSize * 1.5)

    x1 = int(mouthCenter[0, 0] - (mouthSize / 1.8))
    x2 = int(mouthCenter[0, 0] + (mouthSize / 1.8))
    y1 = int(mouthCenter[0, 1] - (mouthSize / 2)*0.5)
    y2 = int(mouthCenter[0, 1] + (mouthSize / 2)*0.5)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    mouthOverlayWidth = x2 - x1
    mouthOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    mouthOverlay = cv2.resize(imgMouth, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_mouth, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_mouth, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(mouthOverlay, mouthOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Open Mouth
def mouth_open_size(mouth):
    mouthWidth = dist.euclidean(mouth[0], mouth[6])
    hull = ConvexHull(mouth)
    mouthCenter = np.mean(mouth[hull.vertices, :], axis=0)

    mouthCenter = mouthCenter.astype(int)

    distMouth = mouth[18, 1] - mouth[14, 1]

    return int(mouthWidth), mouthCenter, distMouth

def place_mouth_open(frame, mouthCenter, mouthSize):
    mouthSize = int(mouthSize * 1.5)

    x1 = int(mouthCenter[0, 0] - (mouthSize / 1.5))
    x2 = int(mouthCenter[0, 0] + (mouthSize / 1.5))
    y1 = int(mouthCenter[0, 1] - (mouthSize / 1.8)*0.7)
    y2 = int(mouthCenter[0, 1] + (mouthSize / 1.8)*0.7)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    mouthOverlayWidth = x2 - x1
    mouthOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    mouthOverlay = cv2.resize(imgMouthOpen, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_mouth_open, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_mouth_open, (mouthOverlayWidth, mouthOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(mouthOverlay, mouthOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Masks
def masks_size(masks):
    masksWidth = dist.euclidean(masks[0], masks[15])
    masksCenter = masks[29]

    return int(masksWidth), masksCenter

def place_masks(frame, masksCenter, masksSize):
    masksSize = int(masksSize * 1.5)

    x1 = int(masksCenter[0, 0] - (masksSize / 2))
    x2 = int(masksCenter[0, 0] + (masksSize / 2))
    y1 = int(masksCenter[0, 1] - (masksSize / 2))
    y2 = int(masksCenter[0, 1] + (masksSize / 2))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    masksOverlayWidth = x2 - x1
    masksOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    masksOverlay = cv2.resize(imgMasks, (masksOverlayWidth, masksOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_masks, (masksOverlayWidth, masksOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_masks, (masksOverlayWidth, masksOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(masksOverlay, masksOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Nose And Year
def nose_and_year_size(noseandyear):
    noseandyearWidth = dist.euclidean(noseandyear[0], noseandyear[15])
    noseandyearCenter = noseandyear[26]

    return int(noseandyearWidth), noseandyearCenter

def place_noseandyear(frame, noseandyearCenter, noseandyearSize):
    noseandyearSize = int(noseandyearSize * 1.2)

    x1 = int(noseandyearCenter[0, 0] - (noseandyearSize / 2))
    x2 = int(noseandyearCenter[0, 0] + (noseandyearSize / 2))
    y1 = int(noseandyearCenter[0, 1] - (2*noseandyearSize / 3))
    y2 = int(noseandyearCenter[0, 1] + (noseandyearSize / 3))

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    noseandyearOverlayWidth = x2 - x1
    noseandyearOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    noseandyearOverlay = cv2.resize(imgNoseAndYear, (noseandyearOverlayWidth, noseandyearOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_noseandyear, (noseandyearOverlayWidth, noseandyearOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_noseandyear, (noseandyearOverlayWidth, noseandyearOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(noseandyearOverlay, noseandyearOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute Glasses
def glasses_size(glasses):
    glassesWidth = dist.euclidean(glasses[0], glasses[15])
    glassesCenter = glasses[26]

    return int(glassesWidth), glassesCenter

def place_glasses(frame, glassesCenter, glassesSize):
    glassesSize = int(glassesSize * 1.1)

    x1 = int(glassesCenter[0, 0] - (glassesSize / 2))
    x2 = int(glassesCenter[0, 0] + (glassesSize / 2))
    y1 = int(glassesCenter[0, 1] - (2*glassesSize / 3)*0.7)
    y2 = int(glassesCenter[0, 1] + (glassesSize / 3)*0.7)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    glassesOverlayWidth = x2 - x1
    glassesOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    glassesOverlay = cv2.resize(imgGlasses, (glassesOverlayWidth, glassesOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_glasses, (glassesOverlayWidth, glassesOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_glasses, (glassesOverlayWidth, glassesOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(glassesOverlay, glassesOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

#Compute eyebrows
def eyebrows_size(eyebrows):
    eyebrowsWidth = dist.euclidean(eyebrows[0], eyebrows[9])
    eyebrowsCenter = eyebrows[10]

    return int(eyebrowsWidth), eyebrowsCenter

def place_eyebrows(frame, eyebrowsCenter, eyebrowsSize):
    eyebrowsSize = int(eyebrowsSize * 1.5)

    x1 = int(eyebrowsCenter[0, 0] - (eyebrowsSize / 2)*2)
    x2 = int(eyebrowsCenter[0, 0] + (eyebrowsSize / 2)*2)
    y1 = int(eyebrowsCenter[0, 1] * 0.8 - (200 * eyebrowsSize / 201) * 0.9)
    y2 = int(eyebrowsCenter[0, 1] * 0.8 + (eyebrowsSize / 201) * 0.9)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    eyebrowsOverlayWidth = x2 - x1
    eyebrowsOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    eyebrowsOverlay = cv2.resize(imgEyebrows, (eyebrowsOverlayWidth, eyebrowsOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_eyebrows, (eyebrowsOverlayWidth, eyebrowsOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_eyebrows, (eyebrowsOverlayWidth, eyebrowsOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(eyebrowsOverlay, eyebrowsOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

def estimate_distance(pointsFace):

    thirtyseven = pointsFace[36]
    ninteen = pointsFace[18]
    fortyfour = pointsFace[43]
    twentyfour = pointsFace[23]

    face_width = dist.euclidean(pointsFace[15], pointsFace[0])
    dis_right = dist.euclidean(thirtyseven, ninteen)
    dis_left = dist.euclidean(fortyfour, twentyfour)

    return int(face_width), dis_right, dis_left

#Compute stars
def stars_size(stars):
    starsWidth = dist.euclidean(stars[0], stars[9])
    starsCenter = stars[10]

    return int(starsWidth), starsCenter

def place_stars(frame, starsCenter, starsSize):
    starsSize = int(starsSize * 1.5)

    x1 = int(starsCenter[0, 0] - (starsSize / 2)*3)
    x2 = int(starsCenter[0, 0] + (starsSize / 2)*3)
    y1 = int(starsCenter[0, 1] - (starsSize / 2)*3)
    y2 = int(starsCenter[0, 1] + (starsSize / 2)*3)

    h, w = frame.shape[:2]

    # check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # re-calculate the size to avoid clipping
    starsOverlayWidth = x2 - x1
    starsOverlayHeight = y2 - y1
    # calculate the masks for the overlay
    starsOverlay = cv2.resize(imgEffectbrows, (starsOverlayWidth, starsOverlayHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(origin_mask_stars, (starsOverlayWidth, starsOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv_stars, (starsOverlayWidth, starsOverlayHeight), interpolation=cv2.INTER_AREA)

    # take ROI for the verlay from background, equal to size of the overlay image
    roi = frame[y1:y2, x1:x2]

    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image pixels of the overlay only where the overlay should be
    roi_fg = cv2.bitwise_and(starsOverlay, starsOverlay, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    frame[y1:y2, x1:x2] = dst

print("************************************************************************")
flag = False
choose = input('Choose:\n (0) eyes filter,\n (1) nose filter,\n (2) hat filter,\n (3) moustache filter,\n (4) mouth filter,\n (5) mask filter,\n (6) nose and ears filter,\n (7) glasses filter,\n (8) nose and eyes filter,\n (9) moustache and glasses filter,\n (10) eyebrows filter: ')
while choose > 10:
    print("Warning: type value less than 11!")
    choose = input('***Choose:\n (0) eyes filter,\n (1) nose filter,\n (2) hat filter,\n (3) moustache filter,\n (4) mouth filter,\n (5) mask filter,\n (6) nose and ears filter,\n (7) glasses filter,\n (8) nose and eyes filter,\n (9) moustache and glasses filter,\n (10) eyebrows filter: ')

print("************************************************************************")
print("Type 'q' to close the window!\n Type 'c' to change filter!")
print("************************************************************************")

# ---------------------------------------------------------
# Load and pre-process the eye-overlay
# ---------------------------------------------------------
# Load the image to be used as our overlay
imgEye = cv2.imread('png/Eye.png', -1)
imgNose = cv2.imread('png/Nose.png', -1)
imgHat = cv2.imread('png/Hat.png', -1)
imgMoustache = cv2.imread('png/Moustache.png', -1)
imgMouth = cv2.imread('png/Close-Mouth.png', -1)
imgMouthOpen = cv2.imread('png/Open-Mouth.png', -1)
imgMasks = cv2.imread('png/Mask.png', -1)
imgNoseAndYear = cv2.imread('png/NoseAndYear.png', -1)
imgGlasses = cv2.imread('png/Glasses.png', -1)
imgEyebrows = cv2.imread('png/TextEyebrows.png', -1)
imgEffectbrows = cv2.imread('png/Stars.png', -1)

origin_mask_hat = imgHat[:, :, 3]
orig_mask_inv_hat = cv2.bitwise_not(origin_mask_hat)
imgHat = imgHat[:, :, 0:3]
origHatHeight, origHatWidth = imgHat.shape[:2]

origin_mask_mouth = imgMouth[:, :, 3]
orig_mask_inv_mouth = cv2.bitwise_not(origin_mask_mouth)
imgMouth = imgMouth[:, :, 0:3]
origMouthHeight, origMouthWidth = imgMouth.shape[:2]

origin_mask_mouth_open = imgMouthOpen[:, :, 3]
orig_mask_inv_mouth_open = cv2.bitwise_not(origin_mask_mouth_open)
imgMouthOpen = imgMouthOpen[:, :, 0:3]
origMouthOpenHeight, origMouthOpenWidth = imgMouthOpen.shape[:2]

origin_mask_masks = imgMasks[:, :, 3]
orig_mask_inv_masks = cv2.bitwise_not(origin_mask_masks)
imgMasks = imgMasks[:, :, 0:3]
origMasksHeight, origMasksWidth = imgMasks.shape[:2]

origin_mask_noseandyear = imgNoseAndYear[:, :, 3]
orig_mask_inv_noseandyear = cv2.bitwise_not(origin_mask_noseandyear)
imgNoseAndYear = imgNoseAndYear[:, :, 0:3]
origNoseAndYearHeight, origNoseAndYearWidth = imgMasks.shape[:2]

# Create the mask from the overlay image
orig_mask_eye = imgEye[:, :, 3]
# Create the inverted mask for the overlay image
orig_mask_inv_eye = cv2.bitwise_not(orig_mask_eye)
# Convert the overlay image image to BGR
# and save the original image size
imgEye = imgEye[:, :, 0:3]
origEyeHeight, origEyeWidth = imgEye.shape[:2]

origin_mask_nose = imgNose[:, :, 3]
orig_mask_inv_nose = cv2.bitwise_not(origin_mask_nose)
imgNose = imgNose[:, :, 0:3]
origNoseHeight, origNoseWidth = imgNose.shape[:2]

# Create the mask from the overlay image
origin_mask_glasses = imgGlasses[:, :, 3]
# Create the inverted mask for the overlay image
orig_mask_inv_glasses = cv2.bitwise_not(origin_mask_glasses)
# Convert the overlay image image to BGR
# and save the original image size
imgGlasses = imgGlasses[:, :, 0:3]
origGlassesHeight, origGlassesWidth = imgGlasses.shape[:2]

origin_mask_moustache = imgMoustache[:, :, 3]
orig_mask_inv_moustache = cv2.bitwise_not(origin_mask_moustache)
imgMoustache = imgMoustache[:, :, 0:3]
origMoustacheHeight, origMoustacheWidth = imgMoustache.shape[:2]

#elif choose == 10:
origin_mask_eyebrows = imgEyebrows[:, :, 3]
orig_mask_inv_eyebrows = cv2.bitwise_not(origin_mask_eyebrows)
imgEyebrows = imgEyebrows[:, :, 0:3]
origEyebrowsHeight, origEyebrowsWidth = imgEyebrows.shape[:2]

origin_mask_stars = imgEffectbrows[:, :, 3]
orig_mask_inv_stars = cv2.bitwise_not(origin_mask_stars)
imgEffectbrows = imgEffectbrows[:, :, 0:3]
origStarsHeight, origStarsWidth = imgEyebrows.shape[:2]

# Start capturing the WebCam
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()

            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

            if choose == 0:
                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]
                #center_nose = landmarks[NOSE_POINTS]

                # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                leftEyeSize, leftEyeCenter = eye_size(left_eye)
                rightEyeSize, rightEyeCenter = eye_size(right_eye)
                #centerNoseSize, centerNoseCenter = nose_size(center_nose)

                place_eye(frame, leftEyeCenter, leftEyeSize)
                place_eye(frame, rightEyeCenter, rightEyeSize)
                #place_nose(frame, centerNoseCenter, centerNoseSize)
            elif choose == 1:
                center_nose = landmarks[NOSE_POINTS]
                centerNoseSize, centerNoseCenter = nose_size(center_nose)
                place_nose(frame, centerNoseCenter, centerNoseSize)

            elif choose == 2:
                center_hat = landmarks[FACE_POINTS]
                centerHatSize, centerHatCenter = hat_size(center_hat)
                place_hat(frame, centerHatCenter, centerHatSize)

            elif choose == 3:
                center_moustache = landmarks[MOUTH_OUTLINE_POINTS]
                centerMoustacheSize, centerMoustacheCenter = moustache_size(center_moustache)
                place_moustache(frame, centerMoustacheCenter, centerMoustacheSize)

            elif choose == 4:
                center_mouth = landmarks[MOUTH_POINTS]
                centerMouthSize, centerMouthCenter, distMouth = mouth_size(center_mouth)
                if distMouth < 5:
                    place_mouth(frame, centerMouthCenter, centerMouthSize)
                else:
                    center_mouth_open = landmarks[MOUTH_POINTS]
                    centerMouthOpenSize, centerMouthOpenCenter, distMouth = mouth_open_size(center_mouth_open)
                    place_mouth_open(frame, centerMouthOpenCenter, centerMouthOpenSize)

            elif choose == 5:
                center_masks = landmarks[FULL_POINTS]
                centerMasksSize, centerMasksCenter = masks_size(center_masks)
                place_masks(frame, centerMasksCenter, centerMasksSize)

            elif choose == 6:
                center_noseandyear = landmarks[FULL_POINTS]
                centerNoseAndYearSize, centerNoseAndYearCenter = masks_size(center_noseandyear)
                place_noseandyear(frame, centerNoseAndYearCenter, centerNoseAndYearSize)

            elif choose == 7:
                center_glasses = landmarks[FULL_POINTS]
                centerGlassesSize, centerGlasses = masks_size(center_glasses)
                place_glasses(frame, centerGlasses, centerGlassesSize)

            elif choose == 8:
                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]
                center_nose = landmarks[NOSE_POINTS]

                # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
                leftEyeSize, leftEyeCenter = eye_size(left_eye)
                rightEyeSize, rightEyeCenter = eye_size(right_eye)
                centerNoseSize, centerNoseCenter = nose_size(center_nose)

                place_eye(frame, leftEyeCenter, leftEyeSize)
                place_eye(frame, rightEyeCenter, rightEyeSize)
                place_nose(frame, centerNoseCenter, centerNoseSize)

            elif choose == 9:
                center_glasses = landmarks[FULL_POINTS]
                centerGlassesSize, centerGlasses = masks_size(center_glasses)
                place_glasses(frame, centerGlasses, centerGlassesSize)

                center_moustache = landmarks[MOUTH_OUTLINE_POINTS]
                centerMoustacheSize, centerMoustacheCenter = moustache_size(center_moustache)
                place_moustache(frame, centerMoustacheCenter, centerMoustacheSize)

            elif choose == 10:
                face_width, dis_right, dis_left = estimate_distance(landmarks[FULL_POINTS])
                threshold = (31.4 * face_width) / 152
                if (dis_right > threshold or dis_left > threshold):
                    center_stars = landmarks[FULL_POINTS]
                    centerStarsSize, centerStars = masks_size(center_stars)
                    place_stars(frame, centerStars, centerStarsSize)
                else:
                    center_eyebrows = landmarks[FACE_POINTS]
                    centerEyebrowsSize, centerEyebrows = eyebrows_size(center_eyebrows)
                    place_eyebrows(frame, centerEyebrows, centerEyebrowsSize)


        cv2.imshow("Faces with Overlay", frame)

    ch = 0xFF & cv2.waitKey(1)

    if ch == ord('q'):
        break
    if ch == ord('c'):
        choose = choose + 1
        if choose > 10:
            choose = 0

cv2.destroyAllWindows()
