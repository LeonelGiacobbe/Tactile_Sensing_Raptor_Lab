import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *

def capture_image(dev):
    f0 = dev.get_raw_image()
    if f0 is not None:
        cv2.imwrite(f'image.jpg', f0)
        # print("Image saved")
    else:
        print("Error: No image captured")

def main():
    # Counter for captured frame identification
    counter1 = 1
    counter2 = 1
    # Gelsight connection
    # To define multiple connections, edit gsdevice to accept dev_id as a constructor argument
    # That way we can instantiate multiple objects, according to /dev/videoX
    # dev_id will be the X in videoX
    # For 2f-140 gripper
    dev2 = gsdevice.Camera("GelSight Mini", 4) # second arg should be X in videoX 
    # # For 2f-85 gripper
    # dev1 = gsdevice.Camera("Gelsight Mini", 2) # second arg should be X in videoX

    # dev1.connect()
    dev2.connect()

    capture_image(dev2)

    print("Done capturing image")
        

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball