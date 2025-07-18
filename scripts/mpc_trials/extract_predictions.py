import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def pose_to_homogeneous_matrix(pose_obj):
    rx, ry, rz = np.deg2rad(pose_obj.theta_x), np.deg2rad(pose_obj.theta_y), np.deg2rad(pose_obj.theta_z)
    rot_matrix = R.from_euler('xyz', [rx, ry, rz]).as_matrix()

    H = np.eye(4)
    H[:3, :3] = rot_matrix
    H[0, 3] = pose_obj.x
    H[1, 3] = pose_obj.y
    H[2, 3] = pose_obj.z
    return H

def joint_0_1_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [-np.sin(angle),  -np.cos(angle), 0, 0],
        [0, 0, -1, 0.1564],
        [0, 0, 0, 1]
    ])

def joint_1_2_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, -1, 0.0054],
        [np.sin(angle),  np.cos(angle), 0, -0.1284],
        [0, 0, 0, 1]
    ])

def joint_2_3_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, -1, -0.2104],
        [-np.sin(angle),  -np.cos(angle), 0, -0.0064],
        [0, 0, 0, 1]
    ])

def joint_3_4_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, -1, -0.0064],
        [np.sin(angle),  np.cos(angle), 0, -0.2104],
        [0, 0, 0, 1]
    ])

def joint_4_5_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, -1, -0.2084],
        [-np.sin(angle),  -np.cos(angle), 0, -0.0064],
        [0, 0, 0, 1]
    ])

def joint_5_6_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, -1, 0],
        [np.sin(angle),  np.cos(angle), 0, -0.1059],
        [0, 0, 0, 1]
    ])

def joint_6_7_transform(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [0, 0, 1, -0.1059],
        [-np.sin(angle),  -np.cos(angle), 0, 0],
        [0, 0, 0, 1]
    ])

def get_inverse_transform(transform_matrix):
    return np.linalg.inv(transform_matrix)

def calculate_joint_transforms(original_matrix):
    # Convert from joint 7 to 6
    tr_matrix = get_inverse_transform(joint_6_7_transform(angle=0))

def homogeneous_matrix_to_grasp(matrix):
    print("  Position/rotation matrix at max confidence index (using tool as reference frame):")
    print(matrix)

    print("  Rotation after converting matrix to angles:")
    rotation_obj = R.from_matrix(matrix[:3, :3])
    euler_zyx_rad = rotation_obj.as_euler('zyx')
    target_rotation = np.rad2deg(euler_zyx_rad)
    print(target_rotation)

    print("Position extracted from matrix:")
    target_posi = [matrix[0,2], matrix[1,2], matrix[2,2]]
    print(target_posi)

    res = []
    res.extend(posi.item() for posi in target_posi)
    res.append(0.)
    res.extend(rot.item() for rot in target_rotation)
    return res

try:
    target_posi = None
    target_rotation = None
    npz_keys = np.load('predictions_1.npz', allow_pickle=True)

    print(f"NpzFile with keys: {list(npz_keys.keys())}")
    
    if "scores" in npz_keys.keys() and "pred_grasps_cam" in npz_keys.keys():
        # Get the value associated with the 'scores' key
        # This value is a list containing a dictionary
        scores_data_list = npz_keys["scores"]
        posi_pred_cam_list = npz_keys["pred_grasps_cam"]

        # Check if it's a list and if it contains a dictionary as expected
        if (isinstance(scores_data_list, np.ndarray) and scores_data_list.ndim == 0 and isinstance(scores_data_list.item(), dict)) and \
           (isinstance(posi_pred_cam_list, np.ndarray) and posi_pred_cam_list.ndim == 0 and isinstance(posi_pred_cam_list.item(), dict)):
            
            # Extract the dictionary from the 0-d array wrapper
            scores_dict = scores_data_list.item()
            posi_pred_cam_dict = posi_pred_cam_list.item()

            print("\nFinding max confidence value, its index, and the posi/rotation:")
            for key, score_array_value in scores_dict.items():
                max_confidence = 0.0
                max_index = -1
                if isinstance(score_array_value, np.ndarray):
                    # Ensure it's at least 1-D for sorting
                    if score_array_value.ndim == 0:
                        score_array_value = np.array([score_array_value.item()])
                    current_max_confidence = np.max(score_array_value)
                    current_max_index = np.argmax(score_array_value)

                    print(f"For Key {key}:")
                    print(f"  Max confidence: {current_max_confidence}")
                    print(f"  Index of max confidence: {current_max_index}")

                    if key in posi_pred_cam_dict:
                        corresponding_grasp_array = posi_pred_cam_dict[key]
                        if isinstance(corresponding_grasp_array, np.ndarray) and corresponding_grasp_array.ndim >= 1:
                            grasp_matrix_at_max_confidence = corresponding_grasp_array[current_max_index]
                            
                            print(f"Res: ", homogeneous_matrix_to_grasp(grasp_matrix_at_max_confidence))

                else:
                    print(f"Warning: Value for key {key} is not a NumPy array. Type: {type(score_array_value)}")

        else:
            print(f"The 'scores' key does not contain the expected list of dictionaries. Type: {type(scores_data_list)}, Dims: {getattr(scores_data_list, 'ndim', 'N/A')}")
            print(f"Content of 'scores' key: {scores_data_list}")

    else:
        print("\n'scores' key not found in the .npz file.")

    npz_keys.close()
    

except FileNotFoundError:
    print("Error: The file was not found. Please check the path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")