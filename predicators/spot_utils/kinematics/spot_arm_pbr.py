import os
import pbrspot
import numpy as np
from predicators.spot_utils.kinematics.spot_arm import SpotArmFK
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from bosdyn.client.math_helpers import SE3Pose, Quat
from scipy.optimize import approx_fprime


class SpotArmPBR:
    def __init__(self, rng, tol=1e-2):
        # DH parameters for Spot arm (approximate values)
        # [theta, d, a, alpha]
        self.robot = pbrspot.spot.Spot()
        self.tol = tol
        self.rng = rng
        # Joint limits (as radians). for whole body IK
        # for arm IK, it is in the Pybullet URDF Link def
        self.joint_limits = [
            [-5*np.pi/6, np.pi],        # shoulder_roll
            [-np.pi, np.pi/6],          # shoulder_pitch
            [0, np.pi],                 # elbow_pitch
            [-8*np.pi/9, 8*np.pi/9],    # elbow_yaw
            [-7*np.pi/12, 7*np.pi/12],  # wrist_pitch
            [-8*np.pi/9, 8*np.pi/9]    # wrist_yaw
        ]

    def compute_fk(self, body_pose, joint_angles):
        """
        Compute forward kinematics given base pose and joint angles.
        base_pose: 4x4 transformation matrix of the base pose
        joint_angles: list of 6 joint angles in radians
        """        
        # Initialize transformation matrix
        self.robot.set_body_transform(body_pose)
        # Compute forward kinematics
        hand_pose = self.robot.arm.ComputeFK(joint_angles)
        return hand_pose
    
    def compute_ik(self, target_hand_pose, body_pose):
        """
        Compute inverse kinematics for the robot arm given a target hand pose.
        If no solution is found within tolerance, return all zeros.
        """
        # first transform the body
        self.robot.set_body_transform(body_pose)
        # Objective function to minimize distance between FK result and target pose
        newq = self.robot.arm.ComputeIK(target_hand_pose, self.rng)
        
        if newq is not None:
            return True, newq
        else:
            return False, [0] * 6
        
    def compute_whole_body_ik(self, target_hand_pose, 
                              body_constraint,
                              init_body_pose=None,
                              ):
        """
        Compute inverse kinematics for the whole body given a target hand pose and body constraint.
        We only use pbrspot to compute the FK here.
        """
        # Objective function to minimize distance between FK result and target pose
        def objective(whole_body_pose):
            body_pose_6d = whole_body_pose[:6]
            joint_angles = whole_body_pose[6:]
            body_pose = np.eye(4)
            body_pose[:3, :3] = R.from_euler('xyz', body_pose_6d[3:]).as_matrix()
            body_pose[:3, 3] = body_pose_6d[:3]

            self.robot.set_body_transform(body_pose)
            # sol = self.robot.arm.ComputeIK(target_hand_pose, self.rng)
            self.robot.arm.SetJointValues(joint_angles)
            current_hand_pose = self.robot.arm.GetEETransform()

            # Calculate position and orientation error
            pos_error = np.linalg.norm(current_hand_pose[:3, 3] - target_hand_pose[:3, 3])
            rot_error = np.linalg.norm(current_hand_pose[:3, :3] - target_hand_pose[:3, :3])

            # Total error as sum of position and orientation errors
            return pos_error + rot_error
        
        # Initial guess for joint angles (start with zero angles)
        if init_body_pose is None:
            initial_guesses = [
                [0] * (6 + 6),
                [0, 0, 0, 0, 0, 0, 
                 0, -3/4*np.pi, 3*np.pi/4, 0, 0, 0],
            ]
        else:
            initial_guesses = [
                init_body_pose + [0] * 6,
                init_body_pose + [0, -3/4*np.pi, 3*np.pi/4, 0, 0, 0],
            ]

        best_solution = None
        best_error = float('inf')

        whole_body_bound = body_constraint + self.joint_limits
        # grad = approx_fprime(initial_guesses[0], objective, epsilon=1e-6)
        # print(f"Gradients: {grad}")
        for guess in initial_guesses:
            result = minimize(objective, guess, bounds=whole_body_bound, method='SLSQP')
            if result.success and result.fun < self.tol:
                if result.fun < best_error:
                    best_error = result.fun
                    best_solution = result.x
        
        if best_solution is not None:
            return True, best_solution
        else:
            return False, [0] * 12
        

# Example usage
if __name__ == "__main__":
    # Launch pybullet
    pbrspot.utils.connect(use_gui=True)
    pbrspot.utils.disable_real_time()
    # update these information from ROS2 service localizer
    body_frame_x = 1.62
    body_frame_y = 0.435
    body_frame_z = 0.595
    body_rot = [-0.0002, -0.0009, -0.997, 0.0235] # qx qy qz qw
    pbrspot.utils.set_camera(180, -10, 2.5, [body_frame_x, body_frame_y, body_frame_z-0.74])
    # Create a rng with a seed so that executions stay
    # consistent between runs.
    rng = np.random.default_rng(0)

    # Create SpotArmPBR object, verify it aligns with SpotArmFK
    robot_pbr = SpotArmPBR(rng)
    robot_armfk = SpotArmFK(visualize=True)
    
    # Set base pose [x, y, z, roll, pitch, yaw]
    # Add floor object 
    curr_path = os.getcwd()
    floor_file = '/home/airlabbw/NeSy/NeuPI/ext/pbrspot/models/short_floor.urdf'
    floor = pbrspot.body.createBody(floor_file)
    floor.set_point([0, 0, body_frame_z-0.84])
    # match the ROS2 service localizer
    robot_pbr.robot.set_body_point([body_frame_x, body_frame_y, body_frame_z])
    robot_pbr.robot.set_quat(body_rot)

    # Visualize body pose
    body_pose = robot_pbr.robot.get_body_transform()
    print("Visualizing body pose! {}".format(pbrspot.geometry.pose_from_tform(body_pose)))
    pbrspot.viz.draw_pose(pbrspot.geometry.pose_from_tform(body_pose), length=0.4, width=5)
    pbrspot.utils.wait_for_user()

    # Get the configuration of the arm
    q = robot_pbr.robot.arm.GetJointValues()
    print(f"Current robot arm joint values: {q}")

    # **** Testing Fixed Body FK/IK ****
    # Compute the forward kinematics
    pose_pbr = robot_pbr.compute_fk(body_pose, q)
    pose_armfk_all = robot_armfk.compute_fk(body_pose, q)
    # robot_armfk.run_interactive(body_pose, q)
    print(f"Pose from PBR: {pbrspot.geometry.pose_from_tform(pose_pbr)}")
    print(f"Pose from ArmFK: {pbrspot.geometry.pose_from_tform(pose_armfk_all[-1])}")
    pbrspot.viz.draw_pose(pbrspot.geometry.pose_from_tform(pose_pbr), length=0.4, width=5)
    pbrspot.utils.wait_for_user()


    # Slighly modify to generate a new pose, test IK
    rel_pose = SE3Pose(0.3, 0.0, 0, Quat(1, 0, 0, 0))
    curr_pose = SE3Pose.from_matrix(pose_pbr)
    new_pose = curr_pose * rel_pose
    new_pose_mat = new_pose.to_matrix()
    print(f"Slightly-modified pose to move to: {pbrspot.geometry.pose_from_tform(new_pose_mat)}")

    # Visualize this new pose
    print("Visualizing modified pose!")
    pbrspot.viz.draw_pose(pbrspot.geometry.pose_from_tform(new_pose_mat), length=0.4, width=5)

    succ, newq = robot_pbr.compute_ik(new_pose_mat, body_pose)
    succ, newq_fk = robot_armfk.compute_ik(body_pose, new_pose_mat)
    print(f"Solution from PBR: {newq}")
    print(f"Solution from ArmFK: {newq_fk}")
    hand_pose_fk = robot_armfk.compute_fk(body_pose, newq)[-1]
    assert newq is not None, "Failed to find IK solution!"
    robot_pbr.robot.arm.SetJointValues(newq)

    print(f"New hand pose in ArmFK: {pbrspot.geometry.pose_from_tform(hand_pose_fk)}")
    # Visualize the new pose
    curr_hand_pose = robot_pbr.compute_fk(body_pose, newq)
    print(f"Current robot hand pose: {pbrspot.geometry.pose_from_tform(curr_hand_pose)}")
    pbrspot.utils.wait_for_user()

    # **** Testing Dynamic Body FK/IK ****
    body_rel_pose = SE3Pose(-0.5, 0.1, 0, Quat(1, 0, 0, 0))
    curr_body_pose = SE3Pose.from_matrix(body_pose)
    new_body_pose = (curr_body_pose * body_rel_pose).to_matrix()
    new_hand_pose = robot_pbr.compute_fk(new_body_pose, newq)
    pose_vec = pbrspot.geometry.pose_from_tform(new_hand_pose)

    print(f"New robot hand pose: {pose_vec}")
    pbrspot.viz.draw_pose(pbrspot.geometry.pose_from_tform(new_hand_pose), length=0.4, width=5)
    pbrspot.utils.wait_for_user()

    body_rel_pose = SE3Pose(0.1, -0.1, 0, Quat(1, 0, 0, 0))
    curr_body_pose = SE3Pose.from_matrix(new_body_pose)
    rel_rot_down = Quat.from_pitch(np.pi/2)
    rel_hand_pose = SE3Pose(0.1, 0.0, 0.0, rel_rot_down)
    new_hand_pose = SE3Pose.from_matrix(new_hand_pose) * rel_hand_pose
    new_body_pose = curr_body_pose * body_rel_pose
    succ, new_q = robot_pbr.compute_ik(new_hand_pose.to_matrix(), new_body_pose.to_matrix())

    assert new_q is not None, "Failed to find IK solution!"
    robot_pbr.robot.arm.SetJointValues(new_q)

    # finally test the whole body IK
    curr_hand_pose = robot_pbr.compute_fk(new_body_pose.to_matrix(), new_q)
    rel_hand_pose = SE3Pose(-0.5, 0.0, 0.3, Quat(1, 0, 0, 0))
    curr_hand_pose = SE3Pose.from_matrix(curr_hand_pose)
    new_hand_pose = curr_hand_pose * rel_hand_pose
    print(f"New robot hand pose: {pbrspot.geometry.pose_from_tform(new_hand_pose.to_matrix())}")
    pbrspot.viz.draw_pose(pbrspot.geometry.pose_from_tform(new_hand_pose.to_matrix()), length=0.4, width=5)
    pbrspot.utils.wait_for_user()
    # This should not have sol
    succ, new_q = robot_pbr.compute_ik(new_hand_pose.to_matrix(), new_body_pose.to_matrix())
    assert not succ, "Should not have IK solution!"
    # now tries to solve the whole body IK
    curr_body_z = new_body_pose.z
    new_hand_x = new_hand_pose.x
    new_hand_y = new_hand_pose.y
    new_hand_roll = new_hand_pose.rot.to_roll()
    new_hand_pitch = new_hand_pose.rot.to_pitch()
    new_hand_yaw = new_hand_pose.rot.to_yaw()

    body_constraint = [
        [-1.0 + new_hand_x, 1.0 + new_hand_x],
        [-1.0 + new_hand_y, 1.0 + new_hand_y],
        [-0.0001 + curr_body_z, 0.0001 + curr_body_z],
        [new_hand_roll-np.pi/12, new_hand_roll+np.pi/12],
        [new_hand_pitch-np.pi/6, new_hand_pitch+np.pi/36],
        [new_hand_yaw-np.pi/12, new_hand_yaw+np.pi/12]
    ]
    body_initial_guess = [
        new_hand_x + 0.8,
        new_hand_y,
        curr_body_z,
        new_hand_roll,
        new_hand_pitch,
        new_hand_yaw
    ]
    # use armfk to compute the whole body IK
    succ, sol = robot_armfk.compute_whole_body_ik(new_hand_pose.to_matrix(), 
                                              body_constraint, 
                                              body_initial_guess)
    assert succ, "Failed to find whole body IK solution!"
    print(f"Whole body IK solution: {sol}")
    # visualize the solution in PBR
    body_pos = sol[:3]
    body_euler = sol[3:6]
    joint_angles = sol[6:]
    # euler 2 matrix
    body_rot = R.from_euler('xyz', body_euler).as_matrix()
    body_pose = np.eye(4)
    body_pose[:3, :3] = body_rot
    body_pose[:3, 3] = body_pos
    hand_pose_fk = robot_armfk.compute_fk(body_pose, joint_angles)[-1]
    print(f"Hand pose from ArmFK: {pbrspot.geometry.pose_from_tform(hand_pose_fk)}")
    # robot_armfk.run_interactive(body_pose, joint_angles)
    robot_pbr.robot.set_body_transform(body_pose)
    robot_pbr.robot.arm.SetJointValues(joint_angles)
    pbrspot.utils.wait_for_user()
    hand_pose_pbr = robot_pbr.compute_fk(body_pose, joint_angles)
    print(f"Hand pose from PBR: {pbrspot.geometry.pose_from_tform(hand_pose_pbr)}")