import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # lengthen lines by scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

def getCenter(pts): # pts should be contours -> use cv2's findContours

    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean) # eigen* not used but getting errors if not declared here    
    # return the center of the object
    print(f"The center of the object was: {(int(mean[0,0]), int(mean[0,1]))}")
    return (int(mean[0,0]), int(mean[0,1]))
 
def getOrientation(pts, img):
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    
    # Store the center of the object
    cntr = getCenter(pts)
    
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
    
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    # # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    # cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    
    return angle
 
# Load the images from Kinova arm
video_capture = cv2.VideoCapture("rtsp://192.168.1.10/color")

# angle and center storing lists
angles = []
centers = []

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    return

# Was the image there?
ret, frame = video_capture.read()

if not ret:
    print("Could not get frame from camera")
    return

# Convert image to grayscale
grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Denoise the image
denoiseGray = cv2.fastNlMeansDenoising(grayFrame, None, h=30, templateWindowSize=7, searchWindowSize=21)

# Convert image to binary
_, bw = cv2.threshold(denoiseGray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Find all the contours in the thresholded image
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv2.contourArea(c)
    
    # filter by size
    if area < 1000 or 50000 < area:
        continue
    
    # Draw each contour only for visualisation purposes
    # cv2.drawContours(frame, contours, i, (0, 0, 255), 2)
    
    # Find the orientation of each shape
    angles.append(getOrientation(c, frame))
    centers.append(getCenter(contours))

# cv2.imshow('Output Image', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
    
# To find major / minor axis
# use fitEllipse from opencv. Returns center of ellipse, major and minor axis length, and rotation
# so to find the points to grab, draw a line of len(minor axis) starting at point "center" and rotate it by "rotation"