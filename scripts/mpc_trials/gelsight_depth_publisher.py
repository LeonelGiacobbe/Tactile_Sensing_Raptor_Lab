import sys
import numpy as np
import cv2
import os
import gsdevice
import gs3drecon

class GelsightDepth:
    """
    A class to handle 3D reconstruction using a GelSight device.
    """

    def __init__(self, save_video=False, find_roi=False, use_gpu=False, mask_markers=False, device_name="GelSight Mini",
                 mmpp=0.0634, net_file_path='nnmini.pt', dev_id=0):
        """
        Initializes the GelSightReconstructor.
        Args:
            save_video (bool): If True, saves the video to a file.
            find_roi (bool): If True, allows the user to select a region of interest.
            use_gpu (bool): If True, uses the GPU for reconstruction.
            mask_markers (bool): If True, masks the markers in the image.
            device_name (str): The name of the GelSight device.
            mmpp (float): millimeters per pixel ratio.
            net_file_path (str): The path to the neural network file.
        """
        self.SAVE_VIDEO_FLAG = save_video
        self.FIND_ROI = find_roi
        self.GPU = use_gpu
        self.MASK_MARKERS_FLAG = mask_markers
        self.path = '.'
        self.mmpp = mmpp
        self.dev_id = dev_id
        self.dev = gsdevice.Camera("Gelsight Mini", self.dev_id)
        self.net_file_path = net_file_path
        self.out = None
        self.nn = None
        self.vis3d = None
        self.kernel = np.ones((55, 55), np.uint8)

    def connect(self):
        """
        Connects to the GelSight device and loads the neural network model.
        This should be called once before calling get_count() multiple times.
        """
        self.dev.connect()
        model_file_path = self.path
        net_path = os.path.join(model_file_path, self.net_file_path)
        print(f'net path = {net_path}')
        gpuorcpu = "cuda" if self.GPU else "cpu"
        self.nn = gs3drecon.Reconstruction3D(self.dev)
        self.nn.load_nn(net_path, gpuorcpu)
        #self.vis3d = gs3drecon.Visualize3D(self.dev.imgh, self.dev.imgw, '', self.mmpp)
        f0 = self.dev.get_raw_image()
        if self.SAVE_VIDEO_FLAG:
            file_path = './3dnnlive.mov'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(file_path, fourcc, 60, (f0.shape[1], f0.shape[0]), isColor=True)
            print(f'Saving video to {file_path}')
        if self.FIND_ROI:
            roi = cv2.selectROI(f0)
            roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            cv2.imshow('ROI', roi_cropped)
            print('Press q in ROI image to continue')
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_count(self):
        """
        Captures a single image, processes it, and returns the pixel count.
        This function should be called repeatedly to get new counts.
        Returns:
            int: The non-zero pixel count from the processed binary image.
        """
        if self.nn is None:
            raise RuntimeError("Model is not loaded. Call connect() first.")
        f1 = self.dev.get_image()
        dm = self.nn.get_depthmap(f1, self.MASK_MARKERS_FLAG)

        # Normalize the depth map to 8-bit for thresholding
        normalized = cv2.normalize(dm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological opening to remove noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        count = np.count_nonzero(binary)
        print("Count: ", count)
        return count

    def visualize(self):
        """
        Displays the live image and 3D visualization.
        """
        if self.nn is None:
            raise RuntimeError("Model is not loaded. Call connect() first.")
        f1 = self.dev.get_image()
        bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
        cv2.imshow('Image', bigframe)
        dm = self.nn.get_depthmap(f1, self.MASK_MARKERS_FLAG)
        self.vis3d.update(dm)
        if self.SAVE_VIDEO_FLAG:
            self.out.write(f1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def stop(self):
        """
        Stops the video stream and closes windows.
        """
        self.dev.stop_video()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example Usage
    reconstructor = GelsightDepth(use_gpu=True)
    try:
        reconstructor.connect()
        while True:
            # Get the count
            count = reconstructor.get_count()
            print(f"Obtained pixel count is {count}")

            # Optionally visualize
            #reconstructor.visualize()
            
    except KeyboardInterrupt:
        print('Interrupted!')
    finally:
        reconstructor.stop()