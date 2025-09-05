import csv
import os, time, sys
import gsdevice

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

from newGripperCommand import GripperCommand
from gelsight_depth_publisher import GelsightDepth

# percentage to mm helper functions
def percentage_to_85_opening(percentage):
    return 85 * (100.0 - percentage) / 100.0

def opening_to_85_percentage(opening_mm):
    percentage = opening_mm * 100 / 85

    return 100 - percentage

class SingleAgentPd():

    def __init__(self, gripper):
        
        # Model variables
        self.gripper_posi = 0. # in mm
        self.gripper_posi_mm = 0.
        self.gripper_ini_flag = False
        self.contact_area = 0 # int, thresholded depth image
        self.freq = 15
        self.del_t = 1/self.freq

        self.gelsight_depth = GelsightDepth(use_gpu=True)
        self.gelsight_depth.connect()

        # Gripper object (declared outside of this class in main)
        self.gripper = gripper
    
    def _update_grippers_state(self):
        """
        Polls the latest feedback from the gripper control threads.
        Updates self.gripper_posi_X and self.gripper_vel_X.
        """
        feedback = self.gripper.get_latest_feedback()
        if feedback:
            self.gripper_posi, self.gripper_vel = feedback
            # Convert percentage position to mm for PD input if needed.
            # _run_inference expects mm, so this conversion should happen here or before _run_inference.
            self.gripper_posi_mm = percentage_to_85_opening(self.gripper_posi)
        else:
            raise ValueError("No information in gripper 1 control thread")

    def _send_gripper_commands(self, target_1_percentage):
        """
        Sends non-blocking target commands to both grippers.
        """
        self.gripper.set_target_position_percentage(target_1_percentage)
        # print(f"Sent commands: G1 -> {target_1_percentage:.2f}%, G2 -> {target_2_percentage:.2f}%")
    
    def _run_model(self):
        """
        Main PD loop that continuously gets sensor data, runs inference,
        and sends new commands.
        """
        q_d = 2
        c_ref = 26500
        k_p= 1/40000
        k_d = 1/6000
        # Initial move to start positions (still using set_target_position_percentage for non-blocking)
        initial_target_g1_percentage = opening_to_85_percentage(65.0)

        # Send new commands to grippers (non-blocking)
        self._send_gripper_commands(initial_target_g1_percentage)
        print(f"Sent grippers to initial target positions: G1->{initial_target_g1_percentage} mm")
        print("Gripper at initial positions. Starting PD loop.")
        time.sleep(2)

        loop_dt = 1.0 / self.freq # Calculate desired loop delay
        try:
            graph_timer = time.time()
            csv_write_time = time.time()

            # while not self.gripper_ini_flag:
            #     print("Gripper has not received information to initialize")

            last_contact_area = 0
            man_posi = 0
            while True:
                
                loop_start_time = time.time()

                # Before updating, get current gripper posi to know what sign to use in velocity
                self.prev_gripper_posi_1 = self.gripper_posi_mm
                # Get latest state from grippers (non-blocking)
                self._update_grippers_state()
                self.contact_area = self.gelsight_depth.get_count()
                if self.contact_area > 44000:
                    continue
                if man_posi == 0:
                    man_posi = self.gripper_posi_mm

                target_posi = man_posi + (self.contact_area - (c_ref+(q_d*0)))*k_p + (self.contact_area - last_contact_area)*k_d
                print(f"Current gripper posi: {man_posi}, target posi: {target_posi}")
                
                last_contact_area = self.contact_area
                man_posi = target_posi

                target_pos_percentage = opening_to_85_percentage(target_posi)

                # Send new commands to grippers (non-blocking)
                self._send_gripper_commands(target_pos_percentage)
                
                with open("pd_position_log.csv", mode='a') as f:
                    writer = csv.writer(f)
                    if time.time() - csv_write_time > 0.33:
                        elapsed_time = time.time() - graph_timer
                        writer.writerow([target_posi, elapsed_time])
                        csv_write_time = time.time()
                
                # Enforce PD loop frequency
                loop_end_time = time.time()
                time_elapsed = loop_end_time - loop_start_time
                time_to_sleep = loop_dt - time_elapsed
                if time_to_sleep > 0:
                    #print(f"PD loop time elapsed: {time_elapsed}")
                    time.sleep(time_to_sleep)
                else:
                    print(f"Warning: PD loop is running slower than desired frequency! {time_elapsed:.4f}s vs {loop_dt:.4f}s")           

        except KeyboardInterrupt:
            self.gripper.Cleanup()

        except Exception as e:
            print(f"Critical Error in PD model _run_model: {e}")
            self.gripper.Cleanup()
            import traceback
            traceback.print_exc() 
            return False

        finally:
            # Cleanup should happen in main, as the PD class itself doesn't own the TCP / UDP connections
            print("PD loop terminated.")



def main():
        try:
            os.remove("pd_position_log.csv")
        except:
            pass
        print("Init main function")
    # Initialize arm connection
        import argparse
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities
        print("Imported utilities")

        # Parse arguments
        parser_1 = argparse.ArgumentParser()
        parser_1.add_argument("--proportional_gain", type=float, help="proportional gain used in control loop", default=1.5)
        args1 = utilities.parseConnectionArguments1(parser_1)
        print("parsed arguments")
        with utilities.DeviceConnection.createTcpConnection(args1) as router_1, utilities.DeviceConnection.createUdpConnection(args1) as router_real_time_1:
                print("Established connections to both arms")
                gripper_1 = GripperCommand(router_1, router_real_time_1, args1.proportional_gain, "85", 0)
                gripper_1.start_control_thread()
                print("Created both gripper objects")

                pd_model = SingleAgentPd(gripper_1)
                pd_model._run_model()


if __name__ == "__main__":
    main()
"""
Need:
- function that reads one image from each gelsight
- function that gets current gripper opening, then transform to mm
- function that receives info from above and handles inference
- function that converts back to percentage and commands arms (use threading to do it at the same time)

get posi and vel, run inference, send command, immediately get velocity, wait until Goto is done to send command again?
"""

