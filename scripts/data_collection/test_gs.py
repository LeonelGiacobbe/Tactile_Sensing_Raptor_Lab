import sys, os, time, cv2, gsdevice, glob, shutil, re, random, threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2
from movement_functions import *

def capture_image(dev1, dev2):
    f0 = dev1.get_raw_image()
    f1 = dev2.get_raw_image()
    if f0 is not None:
        cv2.imwrite(f'image.jpg', f0)
        cv2.imwrite(f'image2.jpg', f1)
        return f0
        # print("Image saved")
    else:
        print("Error: No image captured")

def main():
    # Counter for captured frame identification
    counter1 = 1
    counter2 = 1
    
    dev = gsdevice.Camera("GelSight Mini", 0)
    dev2 = gsdevice.Camera("GelSight Mini", 2)

    # dev1.connect()
    dev.connect()
    dev2.connect()

    capture_image(dev, dev2)
        

if __name__ == "__main__":
    main()

# 75.877 makes jenga block fall
# 47.36 for pink ball