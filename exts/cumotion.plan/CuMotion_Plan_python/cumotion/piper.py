import os
from pxr import Sdf, UsdLux
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.types import ArticulationAction

import cumotion
import numpy as np
import statistics

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.abspath(os.path.join(FILE_PATH, "../../../../assets"))
PACKAGE_PATH = os.path.abspath(os.path.join(FILE_PATH, "../../../../package"))

xrdf_path = os.path.join(PACKAGE_PATH, "piper_camera.xrdf")
urdf_path = os.path.join(PACKAGE_PATH, "piper_camera.urdf")



class PiperCuMotionPlanExample():
    def __init__(self):
        self.robot = None
        self.robot_description = None
        self.kinematics = None
        self.end_effector_frame = ""
        self.cumotion_config = None

        # joint_positions 
        self.step = 0
        self.joint_positions_list = []



    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        SingleXFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_light_to_stage()
    
    def load_robot(self):
        """
        This function loads a simple robot onto the stage.  The user can replace this function with code that loads their own robot.
        """
        robot_prim_path = "/World/piper_camera"
        path_to_robot_usd = ASSET_PATH + "/piper_camera.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self.robot = Robot(robot_prim_path)


    def create_target_poses(self):
        """
        This function creates target poses for the robot to move to.  The user can replace this function with code that creates target poses for their own robot.
        """
        poses = []

        # Use constant orientation target with gripper pointing away from Franka along the x-axis
        orientation = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * np.pi)
        # compound rotation to also point gripper downwards along z-axis
        orientation = orientation * cumotion.Rotation3.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.5 * np.pi)
        # Define rectangle on YZ plane (with x offset).
        x = 0.3
        min_y = -0.2
        max_y = 0.2
        min_z = 0.2
        max_z = 0.4

        # Discretize positions along rectangle.
        step = 0.01  # 1 cm between poses.
        for y in np.arange(min_y, max_y, step):
            poses.append(cumotion.Pose3(orientation, np.array([x, y, max_z])))

        for z in np.arange(max_z, min_z, -step):
            poses.append(cumotion.Pose3(orientation, np.array([x, max_y, z])))

        for y in np.arange(max_y, min_y, -step):
            poses.append(cumotion.Pose3(orientation, np.array([x, y, min_z])))

        for z in np.arange(min_z, max_z, step):
            poses.append(cumotion.Pose3(orientation, np.array([x, min_y, z])))

        return poses


    def setup_cumotion(self):
        """This function is meant to be called inside the setup_scene function after the user has loaded their robot.  It sets up the cumotion interface and loads the robot description and kinematics into cumotion.  The user can replace the contents of this function with code that sets up cumotion for their own robot.
        """

        # Load robot description.
        self.robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # Load kinematics.
        self.kinematics = self.robot_description.kinematics()

        # Set end effector frame for Franka.
        self.end_effector_frame = 'link6'

        # Create configuration for inverse kinematics, after first solve the most recent solution will
        # be added to config for use as a c-space seed.
        self.cumotion_config = cumotion.IkConfig()

        print("Successfully set up cumotion with robot description and kinematics")

    def run_ik(self):
        success = True
        translation_errors = []
        orientation_errors = []

        target_poses = self.create_target_poses()
        for target_pose in target_poses:
            results = cumotion.solve_ik(self.kinematics, target_pose, self.end_effector_frame, self.cumotion_config)
        
            print("IK Results: ", results, results.success)
            if not results.success:
                print("IK failed to find a solution for target pose: ", target_pose)
                success = False

            # Check final pose error.
            tool_pose = self.kinematics.pose(results.cspace_position, self.end_effector_frame)
            translation_errors.append(np.linalg.norm(tool_pose.translation - target_pose.translation))
            orientation_errors.append(
                cumotion.Rotation3.distance(tool_pose.rotation, target_pose.rotation))    

            # Add seed to "warm-start" IK to nearby solution.
            self.cumotion_config.cspace_seeds = [results.cspace_position]
            print("cspace position: ", np.rad2deg(results.cspace_position))
            self.joint_positions_list.append(results.cspace_position)

        print("Translation Error:")
        print("  Mean:    ", statistics.mean(translation_errors))
        print("  Median:  ", statistics.median(translation_errors))
        print("  Std Dev: ", statistics.stdev(translation_errors))
        print("Orientation Error:")
        print("  Mean:    ", statistics.mean(orientation_errors))
        print("  Median:  ", statistics.median(orientation_errors))
        print("  Std Dev: ", statistics.stdev(orientation_errors))
        success = success and statistics.median(translation_errors) < 1e-6
        success = success and statistics.median(orientation_errors) < 1e-3

        print("[Cumotion Success]: ", success)
        # print("cumotion config", self.joint_positions_list)

    def update(self, step: float):
        """This function is called on every physics step.  The user can replace the contents of this function with code that they want to run on every physics step."""
        if self.robot is None:
            print("[CuMotion] Robot not loaded. Click Load first.")
            return
        if not self.robot.handles_initialized:
            self.robot.initialize()
            return

        if not self.joint_positions_list:
            print("[CuMotion] No joint positions available. Run IK first.")
            return

        self.step += 1
        joint_pos = self.joint_positions_list[self.step % len(self.joint_positions_list)]
        # add 0 gripper positions to make action the right size
        joint_pos = np.concatenate([joint_pos, np.array([0.0, 0.0])])
        action = ArticulationAction(joint_positions=joint_pos)
        self.robot.apply_action(action)

        

        

