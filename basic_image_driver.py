import cv2

"""
This code will write 'hello.jpg' to the directory the script is run in.
This code runs on Linux.
This code requires the opencv-python package to be installed.
"""

def saveframe(filepath_1, filepath_2):
    # Open Video device using OpenCV2 and V4L2 backend (Linux)
    cap_1 = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Use 0 for the default camera
    cap_2 = cv2.VideoCapture(4, cv2.CAP_V4L2)

    # Check if the camera opened successfully
    if not cap_1.isOpened():
        print("Error: Could not open video cap_1ture device.")
        return

    if not cap_2.isOpened():
        print("Error: Could not open video cap_1ture device.")
        return

    # cap_1ture a frame
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()

    if ret_1:
        # Save the frame to a file
        status_1 = cv2.imwrite(filepath_1, frame_1)
        if status_1:
            print(f"Image saved successfully as {filepath_1}")
        else:
            print(f"Failed to save image {filepath_1}")
    else:
        print("Failed to grab frame.")

    if ret_2:
        # Save the frame to a file
        status_2 = cv2.imwrite(filepath_2, frame_2)
        if status_2:
            print(f"Image saved successfully as {filepath_2}")
        else:
            print(f"Failed to save image {filepath_2}")
    else:
        print("Failed to grab frame.")
    
    # Release the cap_1ture object
    cap_1.release()
    cap_2.release()

if __name__ == '__main__':
    saveframe('cap_1.jpg', 'cap_2.jpg')
