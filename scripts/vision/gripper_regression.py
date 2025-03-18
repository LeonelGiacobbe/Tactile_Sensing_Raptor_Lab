import numpy as np
from sklearn.linear_model import LinearRegression
# This is mainly a helper file used to calculate how big a pixel is in real life cm
# This can help us have a rough estimate of how wide the gripper needs to be to
# grasp an object
training_data = [
    # xPix and yPix are of center of block
    # xPix, yPix, xMov, yMov
    ((730, 172), (9.2, -5.4)), 
    ((427, 291), (6.0, 5.6)), 
    ((934, 92), (11.8, -9.9)), 
    ((928, 242), (7.2, -9.7)), 
    ((778, 125), (10.6, -5.0)), 
    ((622, 117), (11.6, -0.1)), 
    ((386, 328), (4.9, 7.0)), 
    ((577, 318), (5.0, 1.0)), 
    ((693, 184), (9.4, -2.6)), 
    ((700, 106), (11.4, -2.6)),
    ((473, 262), (6.9, 4.2)),  
    ((473, 176), (10.3, 4.3)),
    ((547, 196), (9.2, 2.0)),
    ((904, 175), (9.3, -8.7)),
    ((965, 178), (9.1, -10.6)),
    ((952, 300), (5.4, -10.6)),
    ((373, 380), (3.0, 7.5)),
    ((217, 331), (4.7, 12.3)),
    ((216, 224), (8.3, 12.2)),
    ((319, 83), (12.5, 9.3)),
    ((343, 159), (9.9, 8.6)),
    ((597, 306), (5.4, 0.4)),
    ((597, 175), (9.3, 0.6)),
    ((597, 127), (11.1, 0.5)),
    ((425, 166), (10.6, 5.8)),
    ((356, 272), (6.6, 8.0)),
    ((391, 95), (11.9, 6.9)),
    ((383, 212), (8.4, 7.0)),
    ((401, 128), (10.9, 6.6)),
    ((511, 130), (11.2, 3.1)),
    ((657, 125), (10.7, -1.3)),
    ((225, 289), (4.9, 12.1)),
    ((260, 324), (4.1, 11.1)),
    ((366, 330), (4.6, 7.5)),
    ((360, 390), (2.6, 8.1)),
    ((1006, 305), (5.3, -11.9)),
    ((1023, 231), (7.7, -12.3)),
    ((966, 305), (5.2, -10.3)),
    ((856, 307), (5.9, -7.0)),
    ((921, 253), (7.8, -9.2)),
    ((400, 253), (8.0, 6.7)),
    ((612, 145), (10.3, 0.1)),
    ((890, 190), (9.6, -8.2)),
    ((896, 48), (11.7, -8.4)),   
    ((620, 449), (0.8, 0.0)),
    ((820, 460), (0.0, -6.0)), 
    ((404, 450), (0.1, 6.3)),  
    ((275, 460), (0.7, 10.3)),    
    ((377, 231), (8.7, 7.0)),  
    ((490, 92), (11.8, 3.8)),   
    ((254, 154), (10.2, 11.2)),

]
# Extract data
xPix = np.array([point[0][0] for point in training_data])  # x pixel coordinates
yPix = np.array([point[0][1] for point in training_data])  # y pixel coordinates
xMov = np.array([point[1][0] for point in training_data])  # x movement in cm
yMov = np.array([point[1][1] for point in training_data])  # y movement in cm


# Reshape data for regression
xPix = xPix.reshape(-1, 1)
yPix = yPix.reshape(-1, 1)

# Perform regression for x direction
reg_x = LinearRegression()
reg_x.fit(xPix, xMov)
x_scale = reg_x.coef_[0]  # Scaling factor for x direction (cm/pixel)

# Perform regression for y direction
reg_y = LinearRegression()
reg_y.fit(yPix, yMov)
y_scale = reg_y.coef_[0]  # Scaling factor for y direction (cm/pixel)

# Average the scaling factors for x and y
pixel_to_cm = (x_scale + y_scale) / 2

print(f"Scaling factor (cm/pixel): {pixel_to_cm}")