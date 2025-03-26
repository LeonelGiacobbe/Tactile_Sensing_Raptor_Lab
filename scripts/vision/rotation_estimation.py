import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
 
pixel_to_cm_factor = 0.0064597418580056515
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
    
    rad_angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians

    deg_angle = np.rad2deg(rad_angle)
    
    # # Label with the rotation angle
    # label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    # textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    # cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    
    return deg_angle
 
def findGripperWidth(minor_axis_length):
    # Returns a value between 0 and 0.8, representing the ideal gripper width 
    # To grasp and object based on its minor axis length
    if minor_axis_length > 2100: # exceeds max width of robotiq gripper, use return for negative value
        return -1.0
    width = minor_axis_length * pixel_to_cm_factor
    min_old = 0
    min_new = 0
    max_old = 2100
    max_new = 0.7
    width = ((width - min_old) / (max_old - min_old)) * (max_new - min_new) + min_new

    return minor_axis_length * pixel_to_cm_factor
# Load the images from Kinova arm
video_capture = cv2.VideoCapture("rtsp://192.168.1.10/color")

# angle and center storing lists
angles = []
centers = []

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

#Was the image there?
ret, frame = video_capture.read()


if not ret:
    print("Could not get frame from camera")
    exit()

# Generate region of interest to reduce errors from edges of image
roi_x = 100   # Starting x coordinate
roi_y = 0    # Starting y coordinate
roi_width = 1200  # Width of the ROI
roi_height = 450  # Height of the ROI

# Crop the image using the defined ROI
roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
# Convert image to grayscale
# gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
# # Denoise the image
# denoised_gray = cv2.fastNlMeansDenoising(gray_frame, None, h=30, templateWindowSize=7, searchWindowSize=21)
roi_frame = cv2.convertScaleAbs(roi_frame, alpha=0.75, beta=-20)
# hsv transformation
hsv_image = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the HSV values
lower_bound = np.array([0, 0, 160])
upper_bound = np.array([255, 255, 255])
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
result = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
# Morphological operations to clean the mask
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# Convert image to binary
_, binary_image = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

# Find all the contours in the thresholded image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

minEllipse = [None]*len(contours)

color = (255, 255, 255)

for i, c in enumerate(contours):

    # Calculate the area of each contour
    area = cv2.contourArea(c)
    if c.shape[0] > 5:
        minEllipse[i] = cv2.fitEllipse(c)
    # filter by size
    if area < 500:
        continue
    else:
        minor_axis_length = minEllipse[i][1][0] # used to calculate the desired gripper opening
        print("Minor axis length: ", minor_axis_length)
        # Draw each contour only for visualisation purposes
        cv2.drawContours(roi_frame, contours, i, (0, 0, 255), 2)
        if c.shape[0] > 5:
            cv2.ellipse(roi_frame, minEllipse[i], color, 2)
        
        # Find the orientation of each shape
        angles.append(getOrientation(c, roi_frame))
        print("Angle: ", getOrientation(c, roi_frame))
        centers.append(getCenter(c))

while True:
    cv2.imshow('Output Image', roi_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('Output Image', cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
    
    
# To find major / minor axis
# use fitEllipse from opencv. Returns center of ellipse, major and minor axis length, and rotation
# fitEllipse returns coords of center, then width and height (minor and major axis), and rotation angle of ellipse
# so to find the points to grab, use len of minor axis, at point "center" and rotate gripper by angle amount (see above)