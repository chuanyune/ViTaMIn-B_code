import numpy as np
import cv2
import json
def extract_rotation_translation(transform_matrix):
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3].reshape(3, 1)
    return R, t

def calibrate_hand_eye(ee_2_base_matrices, world_2_quest):
    R1_list = [] 
    t1_list = [] 
    R2_list = []  
    t2_list = []
    for A, B in zip(ee_2_base_matrices, world_2_quest):
        R1, t1 = extract_rotation_translation(A)
        R2, t2 = extract_rotation_translation(B)
        
        R1_list.append(R1) 
        t1_list.append(t1)
        R2_list.append(R2) 
        t2_list.append(t2)
    
    R1_list = [R.astype(np.float64) for R in R1_list]
    t1_list = [t.astype(np.float64) for t in t1_list]
    R2_list = [R.astype(np.float64) for R in R2_list]
    t2_list = [t.astype(np.float64) for t in t2_list]
    
    R_quest_to_ee, t_quest_to_ee = cv2.calibrateHandEye(
        R1_list, t1_list,
        R2_list, t2_list,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS  
    )
    
    quest_to_ee = np.eye(4)
    quest_to_ee[:3, :3] = R_quest_to_ee
    quest_to_ee[:3, 3] = t_quest_to_ee.flatten()
    return quest_to_ee


if __name__ == "__main__":
    ee_2_base_matrices = np.load('./right.npy')
    quest_2_world = np.load('./right_robot.npy')[6:]
    world_2_quest = [np.linalg.inv(matrix) for matrix in quest_2_world]
    assert len(ee_2_base_matrices) == len(world_2_quest)
    print(len(ee_2_base_matrices))
    X = calibrate_hand_eye(ee_2_base_matrices, world_2_quest)
    print("\nCamera to ee transformation matrix:")
    print(X)
    
    np.save('quest_2_ee_right_hand_fix_quest.npy', X)
