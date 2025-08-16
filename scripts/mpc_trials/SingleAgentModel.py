import threading, queue
import torch
from torchvision import transforms
from PIL import Image
from single_agent_functions import ResCNNEncoder, MPClayer
import os, time, sys
import gsdevice, cv2
import torchvision.transforms as transforms
from PIL import Image

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

from newGripperCommand import GripperCommand

# Model architecture params
CNN_hidden1, CNN_hidden2 = 128, 128 
CNN_embed_dim = 20
res_size = 224       
eps = 1e-4
nStep = 15
del_t = 1/25
dropout_p = 0.15

# percentage to mm helper functions
def percentage_to_85_opening(percentage):
    return 85 * (100.0 - percentage) / 100.0

def percentage_to_140_opening(percentage):
    return 140 * (100.0 - percentage) / 100.0

def opening_to_85_percentage(opening_mm):
    percentage = opening_mm * 100 / 85

    return 100 - percentage

def opening_to_140_percentage(opening_mm):
    percentage = opening_mm * 100 / 140

    return 100 - percentage

class SingleAgentMpc():

    def __init__(self, gripper_1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model variables
        self.prev_gripper_posi_1 = 0.0
        self.gripper_posi_1 = 0.0
        self.gripper_vel_1 = 0.0
        self.gripper_posi_1_mm = 0.0
        self.current_image_1 = None

        self.frequency = 25

        # Gelsight devices
        self.dev1 = gsdevice.Camera("Gelsight Mini", 0)
        self.dev1.connect()
        # Warm up camera:
        for i in range(10):
            self.dev1.get_raw_image()
        #time.sleep(5)

        # Conversion of images to tensor
        self.transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0, 0, 0], std=[0.2, 0.2, 0.2])])
        # Neural network stuff
        print(f"Using {self.device} in controller class")
        self.nn_encoder = ResCNNEncoder(hidden1=CNN_hidden1, hidden2=CNN_hidden2, dropP=dropout_p, outputDim=CNN_embed_dim).to(self.device)
        self.mpc_layer = MPClayer(nHidden = CNN_embed_dim, eps = eps, nStep = nStep, del_t = del_t).to(self.device)
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            torch.backends.cudnn.benchmark = True

        # Load weights
        model_path = os.path.join("/home/leo/Documents/LeTac-MPC/LeTac-MPC/v8_checkpoint_epoch_5.pth")
        checkpoint = torch.load(model_path, map_location=torch.device(self.device), weights_only=True)
        self.nn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
        self.mpc_layer.load_state_dict(checkpoint['mpc_layer_state_dict'])
        self.nn_encoder.eval()
        self.mpc_layer.eval()
        print("Loaded MPC weights")

        # Gripper objects (declared outside of this class in main)
        self.gripper_1 = gripper_1

    def _get_gelsight_images(self):
        """
            Request images from both sensors,
            transform to tensors using same procedure as in training
        """
        cv_image_1 = self.dev1.get_raw_image()
        if cv_image_1 is None:
            raise ValueError("Image was not obtained")

        pil_image_1 = Image.fromarray(cv_image_1).convert("RGB")

        self.current_image_1 = self.transform(pil_image_1).to(self.device)

        return self.current_image_1
    
    def _update_grippers_state(self):
        """
        Polls the latest feedback from the gripper control threads.
        Updates self.gripper_posi_X and self.gripper_vel_X.
        """
        feedback1 = self.gripper_1.get_latest_feedback()
        if feedback1:
            self.gripper_posi_1, self.gripper_vel_1 = feedback1
            # Convert percentage position to mm for MPC input if needed.
            # _run_inference expects mm, so this conversion should happen here or before _run_inference.
            self.gripper_posi_1_mm = percentage_to_85_opening(self.gripper_posi_1)
        else:
            raise ValueError("No information in gripper 1 control thread")


    def _send_gripper_commands(self, target_1_percentage):
        """
        Sends non-blocking target commands to both grippers.
        """
        self.gripper_1.set_target_position_percentage(target_1_percentage)
        # print(f"Sent commands: G1 -> {target_1_percentage:.2f}%, G2 -> {target_2_percentage:.2f}%")
        
    def _run_inference(self, image_1, posi_1, vel_1):
        """
            Run inference on cnn encoder, perform pass in MPC layer, return predictions
        """
        with torch.cuda.stream(self.stream), torch.no_grad():
            # Prepare for cnn inference
            image_1 = image_1.unsqueeze(0).to(self.device)
            #start = time.time()
            output_1 = self.nn_encoder(image_1)#.to(self.device) 
            #end = time.time()
            #print("cnn encoder inference time: ", end - start)
            # print("encodings 1: ", output_1)
            # print("encodings 2: ", output_2)

            # Prepare for mpc pass
            posi_1 = torch.tensor([posi_1]).to(self.device)
            vel_1 = torch.tensor([vel_1]).to(self.device)
            #start = time.time()
            pos_sequences_1 = self.mpc_layer(output_1, posi_1, vel_1)
            #end = time.time()
            #print("mpc layer inference time: ", end - start)

        return pos_sequences_1
    
    def _run_model(self):
        """
        Main MPC loop that continuously gets sensor data, runs inference,
        and sends new commands.
        """
        # Initial move to start positions (still using set_target_position_percentage for non-blocking)
        initial_target_g1_percentage = 70.0
        self._send_gripper_commands(initial_target_g1_percentage)
        print(f"Sent grippers to initial target positions: G1->{initial_target_g1_percentage}%")
        
        # Wait until grippers are approximately at initial position before starting MPC loop
        # This is a blocking wait but only happens once at the beginning.
        # print("Waiting for grippers to reach initial positions...")
        # while not (self.gripper_1.is_target_position_reached() and self.gripper_2.is_target_position_reached()):
        #      time.sleep(0.1) # Wait briefly
        print("Grippers at initial positions. Starting MPC loop.")
        time.sleep(2)

        loop_dt = 1.0 / self.frequency # Calculate desired loop delay
        try:
            while True:
                loop_start_time = time.time()

                # Before updating, get current gripper posi to know what sign to use in velocity
                self.prev_gripper_posi_1 = self.gripper_posi_1_mm
                # Get latest state from grippers (non-blocking)
                self._update_grippers_state()
                posi_1_mm_for_mpc = self.gripper_posi_1_mm 
                vel_1_for_mpc = self.gripper_vel_1
                

                if self.prev_gripper_posi_1 > posi_1_mm_for_mpc: # gripper is closing, flip vel to negative as that's what the MPC layer expects
                    vel_1_for_mpc *= -1

                # Get Gelsight images
                image_1 = self._get_gelsight_images()
                
                print(f"Obtained current state: G1 P: {posi_1_mm_for_mpc:.2f}mm, V: {vel_1_for_mpc:.2f}mm/s")
                # print("Image 1 shape:", image_1.shape)
                # print("Image 2 shape:", image_2.shape)

                # Run MPC Inference
                if image_1 is not None:
                    pos_sequences_1_mm = self._run_inference(
                        image_1,
                        posi_1_mm_for_mpc, vel_1_for_mpc
                    )
                else:
                    raise ValueError ("empty gelsight images in run loop")
                #print("Successful inference.")
                print("Target pos sequence 1 (mm): ", pos_sequences_1_mm)

                # Extract first predicted target and convert to percentage (what GripperCommand expects)
                target_pos_1_mm = pos_sequences_1_mm[:, 0].item()

                target_pos_1_percentage = opening_to_85_percentage(target_pos_1_mm)

                # Send new commands to grippers (non-blocking)
                self._send_gripper_commands(target_pos_1_percentage)
                
                # Set images to None again to force update of gelsights
                self.current_image_1 = None
                
                # Enforce MPC loop frequency
                loop_end_time = time.time()
                time_elapsed = loop_end_time - loop_start_time
                time_to_sleep = loop_dt - time_elapsed
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    print(f"Warning: MPC loop is running slower than desired frequency! {time_elapsed:.4f}s vs {loop_dt:.4f}s")           

        except KeyboardInterrupt:
            self.gripper_1.Cleanup()

        except Exception as e:
            print(f"Critical Error in MPC model _run_model: {e}")
            self.gripper_1.Cleanup()
            import traceback
            traceback.print_exc() 
            return False

        finally:
            # Cleanup should happen in main, as the MPC class itself doesn't own the TCP / UDP connections
            print("MPC loop terminated.")


def main():
        print("Init main function")
    # Initialize arm connection
        import argparse
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities

        # Parse arguments
        parser_1 = argparse.ArgumentParser()
        parser_1.add_argument("--proportional_gain", type=float, help="proportional gain used in control loop", default=3.0)
        args1 = utilities.parseConnectionArguments1(parser_1)
        print("Parsed arguments")

        with utilities.DeviceConnection.createTcpConnection(args1) as router_1:
            with utilities.DeviceConnection.createUdpConnection(args1) as router_real_time_1:
                print("Established connections to both arms")
                gripper_1 = GripperCommand(router_1, router_real_time_1, args1.proportional_gain, "85", 0)
                gripper_1.start_control_thread()
                print("Created both gripper objects")

                mpc_model = SingleAgentMpc(gripper_1)
                mpc_model._run_model()


if __name__ == "__main__":
    main()

