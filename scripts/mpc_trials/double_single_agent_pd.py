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

def percentage_to_140_opening(percentage):
    return 140 * (100.0 - percentage) / 100.0

def opening_to_140_percentage(opening_mm):
    percentage = opening_mm * 100 / 140

    return 100 - percentage

class DoubleSingleAgentPd():

    def __init__(self, gripper_1, gripper_2):
        
        # Model variables
        self.gripper_posi_1 = 0. # in mm
        self.gripper_posi_2 = 0. # in mm
        self.gripper_posi_1_mm = 0.
        self.gripper_posi_2_mm = 0.
        self.gripper_ini_flag = False
        self.contact_area_1 = 0 # int, thresholded depth image
        self.contact_area_2 = 0 # int, thresholded depth image
        self.freq = 12
        self.del_t = 1/self.freq

        self.gelsight_depth_1 = GelsightDepth(use_gpu=True, dev_id=0)
        self.gelsight_depth_1.connect()

        self.gelsight_depth_2 = GelsightDepth(use_gpu=True, dev_id=2)
        self.gelsight_depth_2.connect()

        # Gripper object (declared outside of this class in main)
        self.gripper_1 = gripper_1
        self.gripper_2 = gripper_2
    
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

        feedback2 = self.gripper_2.get_latest_feedback()
        if feedback2:
            self.gripper_posi_2, self.gripper_vel_2 = feedback2
            self.gripper_posi_2_mm = percentage_to_140_opening(self.gripper_posi_2)
        else:
            raise ValueError("No information in gripper 2 control thread")

    def _send_gripper_commands(self, target_1_percentage, target_2_percentage):
        """
        Sends non-blocking target commands to both grippers.
        """
        self.gripper_1.set_target_position_percentage(target_1_percentage)
        self.gripper_2.set_target_position_percentage(target_2_percentage)
        # print(f"Sent commands: G1 -> {target_1_percentage:.2f}%, G2 -> {target_2_percentage:.2f}%")
        
    def _run_model(self):
        """
        Main PD loop that continuously gets sensor data, runs inference,
        and sends new commands.
        """
        q_d = 2
        c_ref_1 = 50500 # Might have to use two different c_ref values
        c_ref_2 = 50550 # Might have to use two different c_ref values
        k_p= 1/80000
        k_d = 1/360000
        # Initial move to start positions (still using set_target_position_percentage for non-blocking)
        

        loop_dt = 1.0 / self.freq # Calculate desired loop delay
        try:
            
            csv_write_time = time.time()

            # while not self.gripper_ini_flag:
            #     print("Gripper has not received information to initialize")

            last_contact_area_1 = 0
            man_posi_1 = 0
            last_contact_area_2 = 0
            man_posi_2 = 0

            self.contact_area_1 = self.gelsight_depth_1.get_count()
            self.contact_area_2 = self.gelsight_depth_2.get_count()
            while self.contact_area_1 > 40000 or self.contact_area_2 > 40000:
                self.contact_area_1 = self.gelsight_depth_1.get_count()
                self.contact_area_2 = self.gelsight_depth_2.get_count()

            initial_target_g1_percentage = opening_to_85_percentage(60.0)
            initial_target_g2_percentage = opening_to_140_percentage(60.0)

            # Send new commands to grippers (non-blocking)
            self._send_gripper_commands(initial_target_g1_percentage, initial_target_g2_percentage)
            print(f"Sent grippers to initial target positions: G1->{initial_target_g1_percentage} mm, G2->{initial_target_g2_percentage} mm")
            print("Grippers at initial positions. Starting PD loop.")
            time.sleep(2)
            
            graph_timer = time.time()
            while True:
                
                loop_start_time = time.time()

                # Before updating, get current gripper posi to know what sign to use in velocity
                self.prev_gripper_posi_1 = self.gripper_posi_1_mm
                self.prev_gripper_posi_2 = self.gripper_posi_2_mm
                # Get latest state from grippers (non-blocking)
                self._update_grippers_state()
                self.contact_area_1 = self.gelsight_depth_1.get_count()
                self.contact_area_2 = self.gelsight_depth_2.get_count()
                if man_posi_1 == 0:
                    man_posi_1 = self.gripper_posi_1_mm
                if man_posi_2 == 0:
                    man_posi_2 = self.gripper_posi_2_mm

                target_posi_1 = man_posi_1 + (self.contact_area_1 - (c_ref_1+(q_d*0)))*k_p + (self.contact_area_1 - last_contact_area_1)*k_d
                target_posi_2 = man_posi_2 + (self.contact_area_2 - (c_ref_2+(q_d*0)))*k_p + (self.contact_area_2 - last_contact_area_2)*k_d
                print(f"Current gripper posi 1: {man_posi_1}, target posi 1: {target_posi_1}")
                print(f"Current gripper posi 2: {man_posi_2}, target posi 2: {target_posi_2}")

                last_contact_area_1 = self.contact_area_1
                man_posi_1 = target_posi_1

                last_contact_area_2 = self.contact_area_2
                man_posi_2 = target_posi_2

                target_pos_percentage_1 = opening_to_85_percentage(target_posi_1)
                target_pos_percentage_2 = opening_to_140_percentage(target_posi_2)

                # Send new commands to grippers (non-blocking)
                self._send_gripper_commands(target_pos_percentage_1, target_pos_percentage_2)
                
                with open("headphone_case_40mm_double_pd_position_log.csv", mode='a') as f:
                    writer = csv.writer(f)
                    if time.time() - csv_write_time > 0.33:
                        elapsed_time = time.time() - graph_timer
                        writer.writerow([target_posi_1, target_posi_2, elapsed_time])
                        csv_write_time = time.time()
                        if elapsed_time > 51.0: # Enforce time limit for data recording
                            print("Reached maximum runtime for trial")
                            self.gripper_1.Cleanup()
                            self.gripper_2.Cleanup()
                            sys.exit(0) # exit succesfully
                
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
            self.gripper_1.Cleanup()
            self.gripper_2.Cleanup()

        except Exception as e:
            print(f"Critical Error in PD model _run_model: {e}")
            self.gripper_1.Cleanup()
            self.gripper_2.Cleanup()
            import traceback
            traceback.print_exc() 
            return False

        finally:
            # Cleanup should happen in main, as the PD class itself doesn't own the TCP / UDP connections
            print("PD loop terminated.")



def main():
        try:
            os.remove("headphone_case_40mm_double_pd_position_log.csv")
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
        parser_2 = argparse.ArgumentParser()
        parser_1.add_argument("--proportional_gain", type=float, help="proportional gain used in control loop", default=1.5)
        parser_2.add_argument("--proportional_gain", type=float, help="proportional gain used in control loop", default=1.5)
        args1 = utilities.parseConnectionArguments1(parser_1)
        args2 = utilities.parseConnectionArguments2(parser_2)
        print("parsed arguments")
        with utilities.DeviceConnection.createTcpConnection(args1) as router_1, utilities.DeviceConnection.createTcpConnection(args2) as router_2:
            with utilities.DeviceConnection.createUdpConnection(args1) as router_real_time_1, utilities.DeviceConnection.createUdpConnection(args2) as router_real_time_2:
                print("Established connections to both arms")
                gripper_1 = GripperCommand(router_1, router_real_time_1, args1.proportional_gain, "85", 0)
                gripper_2 = GripperCommand(router_2, router_real_time_2, args2.proportional_gain, "140", 0)
                gripper_1.start_control_thread()
                gripper_2.start_control_thread()
                print("Created both gripper objects")

                pd_model = DoubleSingleAgentPd(gripper_1, gripper_2)
                pd_model._run_model()


if __name__ == "__main__":
    main()
