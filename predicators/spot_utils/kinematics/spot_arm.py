import logging
import numpy as np
import open3d as o3d
import pickle
from bosdyn.client import math_helpers
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from bosdyn.client.math_helpers import SE3Pose, Quat
# For simulated
DEFAULT_HAND_STOW_ANGLES = [1.03712082e-04, -3.11518478e+00, 3.13274956e+00, 1.57154214e+00, -1.90141201e-02, -1.57168961e+00]
DEFAULT_HAND_HOLDING_ANGLES = [0.1558527158670593, -1.0679405155353672, 1.3354094948097246, 0.002664515178258268, 1.3207200093828653, 0.17407994994706474]
# Pose for the body (relative to the foot).
rot_mat_stair = np.array([[-0.93375213,0.03086331,0.35658718], [-0.02810338,-0.99952153,0.01291956], [0.3568153,0.00204236,0.93417272]])
DEFAULT_STAIR2BODY_ONSTAIR_TF = math_helpers.SE3Pose(
    x=0.542, y=0.0, z=0.343, rot=math_helpers.Quat.from_matrix(rot_mat_stair))
# Pose for the object dumped (relative to the body).
DEFAULT_DUMPED_TF = math_helpers.SE3Pose(
    x=0.8, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2.5))
# Pose for the hand to pick up stair, relative to the stair (only used in task init)
rot_mat = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
DEFAULT_STAIR2HAND_TF = math_helpers.SE3Pose(
    x=0.0, y=0.0, z=0.0, rot=math_helpers.Quat.from_matrix(rot_mat))
DEFAULT_HAND2STAIR_TF = DEFAULT_STAIR2HAND_TF.inverse()

def transform_matrix(theta, d, a, alpha):
    """
    Compute the homogeneous transformation matrix using DH parameters.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

class SpotArmFK:
    def __init__(self, tol=1e-2, visualize=False):
        # DH parameters for Spot arm (approximate values)
        # [theta, d, a, alpha]
        self.dh_params = [
            [0, 0, 0, -np.pi/2],                # shoulder_roll
            [0, 0, 0.3385, 0],                  # shoulder_pitch
            [-np.pi/2, 0, 0.073, -np.pi/2],     # elbow_pitch
            [-np.pi, 0.4033, 0, -np.pi/2],      # elbow_yaw
            [np.pi, 0, 0, -np.pi/2],            # wrist_pitch
            [np.pi/2, 0.11745, 0, np.pi/2]       # hand_replative pose
        ]
        # Joint limits (as radians)
        self.joint_limits = [
            (-5*np.pi/6, np.pi),        # shoulder_roll
            (-np.pi, np.pi/6),          # shoulder_pitch
            (0, np.pi),                 # elbow_pitch
            (-8*np.pi/9, 8*np.pi/9),    # elbow_yaw
            (-7*np.pi/12, 7*np.pi/12),  # wrist_pitch
            (-8*np.pi/9, 8*np.pi/9)     # wrist_yaw
        ]
        self.z_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                        [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1]]
        
        self.tol = tol
        # aligned with Pybullet sim
        self.wrist2hand = np.eye(4)
        self.body2base = np.eye(4)

        hand_rot = Quat.from_yaw(np.pi/2)
        self.body2base[:3, 3] = [0.292, 0.0, 0.188]
        self.wrist2hand[:3, 3] = [0.0, 0.07557, 0.0]  # Adjust as needed
        self.wrist2hand[:3, :3] = hand_rot.to_matrix()
        
        # Initialize visualization
        if visualize:
            self.init_visualization()
        
    def init_visualization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Spot Arm Visualization', width=1024, height=768)

        self.setup_camera()
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])  # Dark gray background
        opt.point_size = 5.0
        opt.line_width = 2.0

    def setup_camera(self):
        """
        Set up the camera view for better visualization
        """
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([1, 1, 1])
        
    def compute_fk(self, body_pose, joint_angles):
        """
        Compute forward kinematics given base pose and joint angles.
        base_pose: 4x4 transformation matrix of the base pose
        joint_angles: list of 6 joint angles in radians
        """        
        # Initialize transformation matrix
        base_pose = body_pose @ self.body2base
        T = base_pose
        transforms = [body_pose, T]
        
        # Compute forward kinematics
        for i, theta in enumerate(joint_angles):
            dh = self.dh_params[i]
            Ti = transform_matrix(theta + dh[0], dh[1], dh[2], dh[3])
            T = T @ Ti
            transforms.append(T)
        
        hand_pose = transforms[-1] @ self.wrist2hand
        transforms.append(hand_pose)
        return transforms
    
    def compute_ik(self, body_pose, target_hand_pose):
        """
        Compute inverse kinematics for the robot arm given a target hand pose.
        If no solution is found within tolerance, return all zeros.
        """
        # Objective function to minimize distance between FK result and target pose
        def objective(joint_angles):
            current_hand_pose = self.compute_fk(body_pose, joint_angles)[-1]
            
            # Calculate position and orientation error
            pos_error = np.linalg.norm(current_hand_pose[:3, 3] - target_hand_pose[:3, 3])
            rot_error = np.linalg.norm(current_hand_pose[:3, :3] - target_hand_pose[:3, :3])
            
            # Total error as sum of position and orientation errors
            return pos_error + rot_error
        
        # Test multiple initial guesses
        initial_guesses = [
            [0] * len(self.dh_params),
            [0, -3/4*np.pi, 3*np.pi/4, 0, 0, 0],
        ]
        
        best_solution = None
        best_error = float('inf')
        
        for guess in initial_guesses:
            result = minimize(objective, guess, bounds=self.joint_limits, method='SLSQP')
            if result.success and result.fun < self.tol:
                if result.fun < best_error:
                    best_error = result.fun
                    best_solution = result.x
        
        if best_solution is not None:
            return True, best_solution
        else:
            return False, [0] * len(self.dh_params)
        
    def compute_whole_body_ik(self, target_hand_pose, 
                              body_constraint,
                              init_body_pose=None,
                              ):
        """
        Compute inverse kinematics for the whole body given a target hand pose and body constraint.
        """
        # Objective function to minimize distance between FK result and target pose
        def objective(whole_body_pose):
            body_pose_6d = whole_body_pose[:6]
            joint_angles = whole_body_pose[6:]
            body_pose = np.eye(4)
            body_pose[:3, :3] = R.from_euler('xyz', body_pose_6d[3:]).as_matrix()
            body_pose[:3, 3] = body_pose_6d[:3]

            current_hand_pose = self.compute_fk(body_pose, joint_angles)[-1]

            # Calculate position and orientation error
            pos_error = np.linalg.norm(current_hand_pose[:3, 3] - target_hand_pose[:3, 3])
            rot_error = np.linalg.norm(current_hand_pose[:3, :3] - target_hand_pose[:3, :3])

            # Total error as sum of position and orientation errors
            return pos_error + rot_error
        
        # Initial guess for joint angles (start with zero angles)
        if init_body_pose is None:
            initial_guesses = [
                [0] * (len(self.dh_params) + 6),
                [0, 0, 0, 0, 0, 0, 
                 0, -3/4*np.pi, 3*np.pi/4, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 
                 0, -3/4*np.pi, 3*np.pi/4, 0, np.pi/3, 0],
            ]
        else:
            initial_guesses = [
                init_body_pose + [0] * len(self.dh_params),
                init_body_pose + [0, -3/4*np.pi, 3*np.pi/4, 0, 0, 0],
                init_body_pose + [0, -3/4*np.pi, 3*np.pi/4, 0, np.pi/3, 0],
            ]

        best_solution = None
        best_error = float('inf')

        whole_body_bound = body_constraint + self.joint_limits
        
        for guess in initial_guesses:
            result = minimize(objective, guess, bounds=whole_body_bound, method='SLSQP')
            if result.success and result.fun < self.tol:
                if result.fun < best_error:
                    best_error = result.fun
                    best_solution = result.x
        
        if best_solution is not None:
            return True, best_solution
        else:
            return False, [0] * (len(self.dh_params) + 6)
        
    def visualize(self, transforms):
        """
        Visualize the robot arm using Open3D with markers for each joint's z-axis.
        """

        assert hasattr(self, 'vis'), "Visualization not initialized. Call init_visualization() first."
        # Clear existing geometry
        self.vis.clear_geometries()

        # Position the world frame at a fixed location far from the arm
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Adjust size as needed
        world_frame.translate([0.5, 0.5, 0])  # Position it 0.5 units along x and y axes, adjust as needed
        self.vis.add_geometry(world_frame)

        # Create coordinate frames and markers for each joint's z-axis
        for i, T in enumerate(transforms):
            if i == 0 or i == len(transforms) - 1:
                size = 0.1 # for body frame and hand frame
            else:
                size = 0.05
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            frame.transform(T)
            self.vis.add_geometry(frame)
            
            # Calculate the position for the z-axis marker by offsetting along the z-axis
            z_axis_marker_pos = T[:3, 3] + T[:3, 2] * 0.05  # Offset by 0.05 units along z-axis
            
            # Create a small sphere as a marker
            z_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            z_marker.translate(z_axis_marker_pos)
            color = self.z_color[i] if i < len(self.z_color) else [0.5, 0.5, 0.5]  # Gray for hand frame
            z_marker.paint_uniform_color(color)  # Color for each z-axis marker
            
            # Add the marker to the visualization
            self.vis.add_geometry(z_marker)
        
        # Create links between joints
        for i in range(len(transforms) - 1):
            start = transforms[i][:3, 3]
            end = transforms[i + 1][:3, 3]
            
            # Create line set for links
            points = np.vstack((start, end))
            lines = np.array([[0, 1]])
            colors = np.array([[1, 0, 0]])  # Red links
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            self.vis.add_geometry(line_set)
        
        # Add a grid for better spatial reference
        grid = self.create_grid(size=1.0, grid_spacing=0.1)
        self.vis.add_geometry(grid)
        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def create_grid(self, size=1.0, grid_spacing=0.1):
        """
        Create a reference grid in the XY plane
        """
        # Create grid points
        x = np.arange(-size, size + grid_spacing, grid_spacing)
        y = np.arange(-size, size + grid_spacing, grid_spacing)
        
        points = []
        lines = []
        point_count = 0
        
        # Create vertical and horizontal lines
        for i in x:
            points.extend([[i, -size, 0], [i, size, 0]])
            lines.append([point_count, point_count + 1])
            point_count += 2
            
        for i in y:
            points.extend([[-size, i, 0], [size, i, 0]])
            lines.append([point_count, point_count + 1])
            point_count += 2
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        
        # Set grid color (light gray)
        colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return line_set
    
    def run_interactive(self, body_pose, joint_angles):
        """
        Run interactive visualization
        """
        transforms = self.compute_fk(body_pose, joint_angles)
        self.visualize(transforms)
        
        logging.info("Interactive visualization window opened.")
        logging.info("Controls:")
        logging.info("  Left click + drag: Rotate")
        logging.info("  Right click + drag: Pan")
        logging.info("  Mouse wheel: Zoom")
        logging.info("Press Ctrl+C in terminal to exit")
        
        try:
            while True:
                self.vis.poll_events()
                self.vis.update_renderer()
        except KeyboardInterrupt:
            logging.info("\nClosing visualization...")
        finally:
            self.close()
    
    def close(self):
        """
        Close the visualization window.
        """
        self.vis.destroy_window()
        

# Example usage
if __name__ == "__main__":
    # Initialize the FK solver
    spot_arm = SpotArmFK(visualize=False)

    # Visualize the map
    # pcd = o3d.io.read_point_cloud("predicators/spot_utils/graph_nav_maps/sqh_final/sqh_final.ply")
    # o3d.visualization.draw_geometries_with_editing([pcd])
    print("Grounding z values (average): -0.55")
    ground_z = -0.55
    # Compute onground poses
    print("Computing onground poses and errors...")
    data_file = "predicators/spot_utils/real_world_data/localizer_on_ground_stowed.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    body_poses = data['body']
    measured_hand_poses = data['hand']
    average_height = []
    average_hand_pose_error = {
        "trans": [],
        "rot": []
    }

    for i in range(len(body_poses)):
        if i < 5:
            continue
        body_pose_se3 = SE3Pose.from_matrix(body_poses[i])
        hand_pose_se3 = SE3Pose.from_matrix(measured_hand_poses[i])
        average_height.append(body_pose_se3.z - ground_z)

        # Compute FK
        sim_hand_pose = spot_arm.compute_fk(body_poses[i], DEFAULT_HAND_STOW_ANGLES)[-1]
        sim_hand_pose_loc = sim_hand_pose[:3, 3]
        hand_pose_loc = measured_hand_poses[i][:3, 3]
        trans_error = np.linalg.norm(sim_hand_pose_loc - hand_pose_loc)

        sim_hand_x_axis = sim_hand_pose[:3, 0]
        hand_x_axis = measured_hand_poses[i][:3, 0]
        # compute the angle between the two vectors
        rot_error = np.arccos(np.dot(sim_hand_x_axis, hand_x_axis) / (np.linalg.norm(sim_hand_x_axis) * np.linalg.norm(hand_x_axis)))
        average_hand_pose_error["rot"].append(rot_error)
        average_hand_pose_error["trans"].append(trans_error)
    print("Average height: ", np.mean(average_height))
    print("Height std: ", np.std(average_height))
    print("Average hand pose error (trans): ", np.mean(average_hand_pose_error["trans"]))
    print("hand pose error std (trans): ", np.std(average_hand_pose_error["trans"]))
    print("Average hand pose error (rot): ", np.mean(average_hand_pose_error["rot"]))
    print("hand pose error std (rot): ", np.std(average_hand_pose_error["rot"]))

    # Compute stair height and pre-grasping relative pose
    print("Computing stair height, pre-grasping, and post dumping relative pose...")
    data_file = "predicators/spot_utils/real_world_data/pickup_place_0.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    for i in range(1, 5):
        data_file = f"predicators/spot_utils/real_world_data/pickup_place_{i}.pkl"
        with open(data_file, "rb") as f:
            extended_data = pickle.load(f)
        data["before_grasp"]['body'].extend(extended_data["before_grasp"]['body'])
        data["before_grasp"]['hand'].extend(extended_data["before_grasp"]['hand'])
        data["after_grasp"]['body'].extend(extended_data["after_grasp"]['body'])
        data["after_grasp"]['hand'].extend(extended_data["after_grasp"]['hand'])
        data["after_pickup"]['body'].extend(extended_data["after_pickup"]['body'])
        data["after_pickup"]['hand'].extend(extended_data["after_pickup"]['hand'])
        data["after_putdown"]['body'].extend(extended_data["after_putdown"]['body'])
        data["after_putdown"]['hand'].extend(extended_data["after_putdown"]['hand'])
    # after grasp hand (stair)
    stair_averaged_height = []
    # before grasp body and after grasp hand (stair), hand2body
    pre_grasping_rel_trans = []
    # pre_grasping_rel_rot = []
    # after pickup body and after pickup hand (stair)
    post_lifting_rel_trans_error = []
    post_lifting_rel_rot_error = []
    # after putdown body and after putdown hand (stair)
    post_dump_rel_trans_error = []
    post_dump_rel_rot_error = []

    for i in range(len(data["before_grasp"]['body'])):
        if i % 10 < 3:
            continue
        body_pose_se3_before_grasp = SE3Pose.from_matrix(data["before_grasp"]['body'][i])
        hand_pose_se3_after_grasp = SE3Pose.from_matrix(data["after_grasp"]['hand'][i])
        stair_pose_se3 = hand_pose_se3_after_grasp * DEFAULT_HAND2STAIR_TF
        stair_averaged_height.append(stair_pose_se3.z - ground_z)
        # check if IK has solution
        succ, _ = spot_arm.compute_ik(body_pose_se3_before_grasp.to_matrix(), hand_pose_se3_after_grasp.to_matrix())
        if not succ:
            print(f"Failed to compute IK for sample {i}")
            continue
        # Compute pre-grasping relative pose, from stair (hand) to body
        beforebody2grasping_hand = body_pose_se3_before_grasp.inverse() * hand_pose_se3_after_grasp
        # grasping_hand2before_body_quant = grasping_hand2before_body.rot
        # grasping_hand2before_body_rot = np.array([grasping_hand2before_body_quant.x, grasping_hand2before_body_quant.y, grasping_hand2before_body_quant.z, grasping_hand2before_body_quant.w])
        pre_grasping_rel_trans.append(beforebody2grasping_hand.to_matrix()[:3, 3])
        # pre_grasping_rel_rot.append(grasping_hand2before_body_rot)
        # Compute post-lifting relative pose, from stair (hand) to body
        body_pose_se3_after_pickup = SE3Pose.from_matrix(data["after_pickup"]['body'][i])
        hand_pose_se3_after_pickup = SE3Pose.from_matrix(data["after_pickup"]['hand'][i])
        sim_hand_pose = spot_arm.compute_fk(body_pose_se3_after_pickup.to_matrix(), DEFAULT_HAND_HOLDING_ANGLES)[-1]
        sim_hand_pose_loc = sim_hand_pose[:3, 3]
        hand_pose_loc = hand_pose_se3_after_pickup.to_matrix()[:3, 3]
        post_lifting_rel_trans_error.append(np.linalg.norm(sim_hand_pose_loc - hand_pose_loc))
        sim_hand_x_axis = sim_hand_pose[:3, 0]
        hand_x_axis = hand_pose_se3_after_pickup.to_matrix()[:3, 0]
        # compute the angle between the two vectors
        post_lifting_rel_rot_error.append(np.arccos(np.dot(sim_hand_x_axis, hand_x_axis) / (np.linalg.norm(sim_hand_x_axis) * np.linalg.norm(hand_x_axis))))
        # Compute post-dumping relative pose, from stair (hand) to body
        body_pose_se3_after_dump = SE3Pose.from_matrix(data["after_putdown"]['body'][i])
        hand_pose_se3_after_dump = SE3Pose.from_matrix(data["after_putdown"]['hand'][i])
        sim_hand_pose = (body_pose_se3_after_dump * DEFAULT_DUMPED_TF).to_matrix()
        sim_hand_pose_loc = sim_hand_pose[:3, 3]
        hand_pose_loc = hand_pose_se3_after_dump.to_matrix()[:3, 3]
        post_dump_rel_trans_error.append(np.linalg.norm(sim_hand_pose_loc - hand_pose_loc))
        sim_hand_x_axis = sim_hand_pose[:3, 0]
        hand_x_axis = hand_pose_se3_after_dump.to_matrix()[:3, 0]
        # compute the angle between the two vectors
        post_dump_rel_rot_error.append(np.arccos(np.dot(sim_hand_x_axis, hand_x_axis) / (np.linalg.norm(sim_hand_x_axis) * np.linalg.norm(hand_x_axis))))
    print("Average stair height: ", np.mean(stair_averaged_height))
    print("Stair height std: ", np.std(stair_averaged_height))
    # we are only interested in the relative translation
    print("Average pre-grasping relative translation: ", np.mean(pre_grasping_rel_trans, axis=0))
    print("Pre-grasping relative translation std: ", np.std(pre_grasping_rel_trans, axis=0))
    # print("Average pre-grasping relative rotation: ", np.mean(pre_grasping_rel_rot, axis=0))

    print("Average post-lifting relative translation error: ", np.mean(post_lifting_rel_trans_error))
    print("Post-lifting relative translation error std: ", np.std(post_lifting_rel_trans_error))
    print("Average post-lifting relative rotation error: ", np.mean(post_lifting_rel_rot_error))
    print("Post-lifting relative rotation error std: ", np.std(post_lifting_rel_rot_error))

    print("Average post-dumping relative translation error: ", np.mean(post_dump_rel_trans_error))
    print("Post-dumping relative translation error std: ", np.std(post_dump_rel_trans_error))
    print("Average post-dumping relative rotation error: ", np.mean(post_dump_rel_rot_error))
    print("Post-dumping relative rotation error std: ", np.std(post_dump_rel_rot_error))
    
    # finally, measue the on_stair body pose and pitch
    print("Computing on_stair body pose and pitch...")
    # first compute the stair pose (the last pickup dump hand pose)
    data_file = "predicators/spot_utils/real_world_data/pickup_place_4.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    stair_pose_trans = []
    # use body pose here
    stair_pose_rot = []
    for i in range(len(data["after_putdown"]['body'])):
        if i % 10 < 3:
            continue
        body_pose_se3_after_dump = SE3Pose.from_matrix(data["after_putdown"]['body'][i])
        hand_pose_se3_after_dump = SE3Pose.from_matrix(data["after_putdown"]['hand'][i])
        stair_pose_trans.append(hand_pose_se3_after_dump.to_matrix()[:3, 3])
        rev_body_pose = body_pose_se3_after_dump * SE3Pose(x=0, y=0, z=0.0, rot=Quat.from_yaw(np.pi))
        stair_pose_rot.append(rev_body_pose.rot.to_yaw())
    mean_trans = np.mean(stair_pose_trans, axis=0)
    mean_rot = np.mean(stair_pose_rot)
    print("Average stair pose translation: ", mean_trans)
    print("Average stair pose rotation: ", mean_rot)
    stair_pose = SE3Pose(mean_trans[0], mean_trans[1], mean_trans[2], Quat.from_yaw(mean_rot))
    # now compute the on_stair body pose
    data_file = "predicators/spot_utils/real_world_data/on_off_stair_0.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    for i in range(1, 10):
        extended_data_file = f"predicators/spot_utils/real_world_data/on_off_stair_{i}.pkl"
        with open(extended_data_file, "rb") as f:
            extended_data = pickle.load(f)
        data["onground"]['body'].extend(extended_data["onground"]['body'])
        data["onground"]['hand'].extend(extended_data["onground"]['hand'])
        data["onstair"]['body'].extend(extended_data["onstair"]['body'])
        data["onstair"]['hand'].extend(extended_data["onstair"]['hand'])
    
    stair2body_trans_err = []
    stair2body_rot_err = [] # euler angles
    for i in range(len(data["onstair"]['body'])):
        if i % 10 < 3:
            continue
        body_pose_se3_onstair = SE3Pose.from_matrix(data["onstair"]['body'][i])
        # body_trans.append(body_pose_se3_onstair.to_matrix()[:3, 3])
        # roll, pitch, yaw = R.from_matrix(body_pose_se3_onstair.to_matrix()[:3, :3]).as_euler('xyz')
        # body_rot.append(np.array([roll, pitch, yaw]))
    # mean_trans = np.mean(body_trans, axis=0)
    # mean_rot = np.mean(body_rot, axis=0)
    # mean_body_pose = SE3Pose(mean_trans[0], mean_trans[1], mean_trans[2], Quat.from_matrix(R.from_euler('xyz', mean_rot).as_matrix()))
    # mean_rel_pose = stair_pose.inverse() * mean_body_pose
    # print("Average on_stair body pose translation: ", mean_rel_pose.to_matrix()[:3, 3])
    # print("Average on_stair body pose rot: ", mean_rel_pose.rot.to_matrix())
        # Compute on_stair body pose
        simulated_body_pose = stair_pose * DEFAULT_STAIR2BODY_ONSTAIR_TF

        stair2body_trans_err.append(np.linalg.norm(simulated_body_pose.to_matrix()[:3, 3] - body_pose_se3_onstair.to_matrix()[:3, 3]))
        sim_body_x_axis = simulated_body_pose.to_matrix()[:3, 0]
        body_x_axis = body_pose_se3_onstair.to_matrix()[:3, 0]
        # compute the angle between the two vectors
        stair2body_rot_err.append(np.arccos(np.dot(sim_body_x_axis, body_x_axis) / (np.linalg.norm(sim_body_x_axis) * np.linalg.norm(body_x_axis))))
    print("Average on_stair body pose translation error: ", np.mean(stair2body_trans_err))
    print("On_stair body pose translation error std: ", np.std(stair2body_trans_err))
    print("Average on_stair body pose rotation error: ", np.mean(stair2body_rot_err))
    print("On_stair body pose rotation error std: ", np.std(stair2body_rot_err))

    # Compute the default stair pose
    print("Computing default stair pose...")
    data_file = "predicators/spot_utils/real_world_data/pickup_place_5.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    for i in range(len(data["after_grasp"]['hand'])):
        if i % 10 < 3:
            continue
        hand_pose_se3_after_grasp = SE3Pose.from_matrix(data["after_grasp"]['hand'][i])
        stair_pose = hand_pose_se3_after_grasp * DEFAULT_HAND2STAIR_TF
        print("Stair pose: ", stair_pose)
        print("Stair pose rot: ", stair_pose.rot.to_yaw())