import csv
import os, time, sys
import gsdevice
from scipy import sparse
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

from newGripperCommand import GripperCommand
from gelsight_depth_publisher import GelsightDepth

import osqp

def vstack_help(vec, n):
    combo = vec.reshape(vec.size,1)
    single  = vec.reshape(vec.size,1)
    for i in range(n-1):
        combo = np.vstack((combo,single))
    return combo

def zeors_hstack_help(vec, n, size_row, size_col):
    combo = vec
    single  = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    for i in range(n-1):
        combo = sparse.hstack((combo,single))
    return combo

def zeors_hstack_help_inverse(vec, n, size_row, size_col):
    end = vec
    single  = sparse.csc_matrix((size_row, size_col), dtype=np.int8)
    combo = single
    for i in range(n-2):
        combo = sparse.hstack((combo,single))
    combo = sparse.hstack((combo,end))
    return combo

def getCS_(C,S_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(15), C)])
    return C_*S_

def getCT_(C,T_):
    C_ = sparse.block_diag([sparse.kron(sparse.eye(15), C)])
    return C_*T_

def b_CT_x0(b_,CT_,x0):
    return b_ - CT_*x0

# percentage to mm helper functions
def percentage_to_85_opening(percentage):
    return 85 * (100.0 - percentage) / 100.0

def opening_to_85_percentage(opening_mm):
    percentage = opening_mm * 100 / 85

    return 100 - percentage

class SingleAgentMpc():

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
            # Convert percentage position to mm for MPC input if needed.
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
        Main MPC loop that continuously gets sensor data, runs inference,
        and sends new commands.
        """
        N = 15
        q_c = 36
        q_v = 1
        q_d = 2
        q_a = 2
        p=5
        c_ref = 36000
        k_c= 36000
        acc_max = 30
        vel_max = 10
        dim = 4

        del_t=1/self.freq


        # Initial move to start positions (still using set_target_position_percentage for non-blocking)
        initial_target_g1_percentage = opening_to_85_percentage(66.0)

        # Send new commands to grippers (non-blocking)
        self._send_gripper_commands(initial_target_g1_percentage)
        print(f"Sent grippers to initial target positions: G1->{initial_target_g1_percentage} mm")
        print("Gripper at initial positions. Starting MPC loop.")
        time.sleep(2)

        loop_dt = 1.0 / self.freq # Calculate desired loop delay
        try:
            graph_timer = time.time()
            csv_write_time = time.time()

            # while not self.gripper_ini_flag:
            #     print("Gripper has not received information to initialize")
            x_state = np.array([0.,0.,0.,0.])

            # state and control Initialization
            x_state = np.array([0.,0.,0.,0.])
            u0 =np.array([[0.]])

            # reference to track
            r = np.array([c_ref,0,0,0]) 
            r_ = vstack_help(r,N)

            # model
            Ad = sparse.csc_matrix([
            [1,   0,    0,  k_c*del_t],
            [0,   1,    0,          0],
            [0,   0,    1,     -del_t],
            [0,   0,    0,          1]
            ])

            Bd = sparse.csc_matrix([
            [0],
            [0],
            [-0.5*del_t*del_t],
            [del_t]
            ])

            # weights
            Q = sparse.csc_matrix([
            [q_c, q_c*q_d, 0, 0],
            [q_c*q_d, q_c*(q_d**2), 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, q_v]
            ])
            R =q_a*sparse.eye(1)
            QN = p*Q
            Q_ = sparse.block_diag([sparse.kron(sparse.eye(N-1), Q), QN], format='csc')
            R_ = sparse.block_diag([sparse.kron(sparse.eye(N), R)], format='csc')

            # T initialization
            T_ = Ad
            temp = Ad
            for i in range(N-1):
                temp = temp.dot(Ad)
                T_ = sparse.vstack([T_,temp])

            I = sparse.eye(dim)
            row_single = zeors_hstack_help(I, N, dim, dim)
            AN_ = row_single
            for i in range(N-1):
                AN = I
                row_single = I
                for j in range(i+1):
                    AN = Ad.dot(AN)
                    row_single = sparse.hstack([AN,row_single])
                row_single = zeors_hstack_help(row_single, N-i-1, dim, dim)
                AN_=sparse.vstack([AN_, row_single])

            Bd_ = sparse.block_diag([sparse.kron(sparse.eye(N), Bd)])
            S_ = AN_*Bd_ 

            # vel and acc constraints
            max_con_b = (np.array([vel_max])).reshape(1,1)
            min_con_b = (np.array([-vel_max])).reshape(1,1)
            u_max = acc_max*np.ones(1*N)
            u_max = u_max.reshape(1*N,1)

            max_con_b_ = vstack_help(max_con_b,N)
            min_con_b_ = vstack_help(min_con_b,N)

            # vel selct matrix
            C_con = sparse.csc_matrix([
            [0,0,0,1]
            ])

            C_con_T_ = getCT_(C_con,T_)

            # real-time vel bounds
            max_con_b_update = b_CT_x0(max_con_b_,C_con_T_,x_state.reshape(dim,1))
            min_con_b_update = b_CT_x0(min_con_b_,C_con_T_,x_state.reshape(dim,1))

            u_ = np.vstack([u_max,max_con_b_update])
            l_ = np.vstack([u_max*-1,min_con_b_update])

            # select matrix for cost function
            L = sparse.eye(dim)
            L_ = sparse.block_diag([sparse.kron(sparse.eye(N), L)], format='csc')

            # QP setup
            P_=2*(R_+(S_.T)*(L_.T)*Q_*L_*S_)
            q_ = 2*(x_state.reshape(1,dim)*(T_.T)*(L_.T)-r_.T)*Q_*L_*S_
            A_=sparse.vstack([sparse.block_diag([sparse.eye(1*N)], format='csc'),getCS_(C_con,S_) ])

            prob = osqp.OSQP()
            prob.setup(P_, q_.T, A_, l_, u_, warm_start=True, max_iter = 8000, verbose=False)
            last_contact_area = 0
            man_posi = 0

            while True:
                loop_start_time = time.time()
                # Get latest state from grippers (non-blocking)
                self._update_grippers_state()
                self.contact_area = self.gelsight_depth.get_count()
                if self.contact_area > 44000:
                    continue

                if x_state[2] == 0:
                    man_posi = self.gripper_posi_mm
                    x_state = np.array([self.contact_area, 0, man_posi, x_state[3]])
                else:
                    x_state = np.array([self.contact_area, 0, x_state[2], x_state[3]])

                # constraints update
                max_con_b_update = b_CT_x0(max_con_b_,C_con_T_,x_state.reshape(dim,1))
                min_con_b_update = b_CT_x0(min_con_b_,C_con_T_,x_state.reshape(dim,1))
                u_ = np.vstack([u_max,max_con_b_update])
                l_ = np.vstack([u_max*-1,min_con_b_update])

                # QP update
                q_ = 2*(x_state.reshape(1,dim)*(T_.T)*(L_.T)-r_.T)*Q_*L_*S_
                prob.update(q=q_.T, l=l_, u=u_)
                res = prob.solve()
                ctrl = res.x[0:1].copy()

                if ctrl[0] is not None:
                    # p, v update
                    x_state = Ad.dot(x_state) + Bd.dot(ctrl)
                print(f"Current gripper posi: {man_posi}, target posi: {x_state[2]}")
                man_posi = x_state[2]
                posi_to_write = man_posi
                
                target_pos_percentage = opening_to_85_percentage(man_posi)

                # Send new commands to grippers (non-blocking)
                self._send_gripper_commands(target_pos_percentage)
                
                with open("mpc_position_log.csv", mode='a') as f:
                    writer = csv.writer(f)
                    if time.time() - csv_write_time > 0.33:
                        elapsed_time = time.time() - graph_timer
                        writer.writerow([self.gripper_posi_mm, elapsed_time])
                        csv_write_time = time.time()
                
                # Enforce MPC loop frequency
                loop_end_time = time.time()
                time_elapsed = loop_end_time - loop_start_time
                time_to_sleep = loop_dt - time_elapsed
                if time_to_sleep > 0:
                    #print(f"MPC loop time elapsed: {time_elapsed}")
                    time.sleep(time_to_sleep)
                else:
                    print(f"Warning: MPC loop is running slower than desired frequency! {time_elapsed:.4f}s vs {loop_dt:.4f}s")           

        except KeyboardInterrupt:
            self.gripper.Cleanup()

        except Exception as e:
            print(f"Critical Error in MPC model _run_model: {e}")
            self.gripper.Cleanup()
            import traceback
            traceback.print_exc() 
            return False

        finally:
            # Cleanup should happen in main, as the MPC class itself doesn't own the TCP / UDP connections
            print("MPC loop terminated.")



def main():
        try:
            os.remove("mpc_position_log.csv")
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

                MPC_model = SingleAgentMpc(gripper_1)
                MPC_model._run_model()


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

