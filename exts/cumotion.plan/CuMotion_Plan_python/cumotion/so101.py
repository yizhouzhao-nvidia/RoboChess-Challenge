import os
import carb 

from pxr import Sdf, UsdLux
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.objects import GroundPlane
from isaacsim.core.simulation_manager import PhysxScene


import cumotion
import numpy as np
import yaml
import math
import statistics

from .chess_utils import FILE_PATH, ASSET_PATH, PACKAGE_PATH,  \
    quaternion_multiply,  \
    add_one_chess_piece, get_piece_pick_up_pose, add_chessboard \
    


class SO101ExampleScript:
    def __init__(self):
        # cumotion
        self.robot_description = None
        self.kinematics = None
        self.end_effector_frame = None

        # articulation
        self.robot: Robot = None
        self.target = None
        self._script_generator = None

        # simulation
        self.step = 0

        # active motion path (list of joint configs from motion planner)
        self._active_path = None
        self._path_index = 0
        self._waypoint_tick = 0    # counts physics steps between waypoints
        self.waypoint_step_interval = 3  # execute one waypoint every N physics steps

        # gripper target position (set by gripper_action; None means not waiting)
        self._gripper_target = None


    def cumotion_setup(
        self, 
        xrdf_path = os.path.join(ASSET_PATH, 'so101', 'lerobot.xrdf'),
        urdf_path = os.path.join(ASSET_PATH, 'so101', 'TheRobotStudio', 'so101_new_calib.urdf'),
        planning_config_path = os.path.join(ASSET_PATH, 'so101', 'lerobot_planner_config.yaml')
    ):
        self.robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
        self.kinematics = self.robot_description.kinematics()
        self.end_effector_frame = self.robot_description.tool_frame_names()[0]

        self.default_cspace = self.robot_description.default_cspace_configuration()
        self.default_tool_pose = self.kinematics.pose(self.default_cspace, self.end_effector_frame)

        self.world_view = cumotion.create_world().add_world_view()

        self.cumotion_config = cumotion.create_motion_planner_config_from_file(
            planning_config_path,
            self.robot_description,
            self.end_effector_frame,
            self.world_view
        )

        self.ik_config = cumotion.create_default_collision_free_ik_solver_config(
            self.robot_description, 
            self.end_effector_frame, 
            self.world_view
        )

        self.ik_solver = cumotion.create_collision_free_ik_solver(self.ik_config)

        self.planner = cumotion.create_motion_planner(self.cumotion_config)

        print("[SO101ExampleScript] cumotion_setup completed.")
            

    def load_example_assets(self, 
                            scenaio: str = "8x8 chess"): #"1D chess"
        # add physics scene
        
        # Create or get the physics scene at this USD path
        physics_scene = PhysxScene("/World/physicsScene")

        # Set physics step to 30 steps/second (dt = 1/30 ≈ 0.0333s)
        physics_scene.set_dt(1.0 / 30.0)

        # add ground plane
        # In setup_scene_fn, after World is created:
        ground = GroundPlane(
            prim_path="/World/GroundPlane",
            z_position=0,          # height in meters
            size=10,             # edge length (optional)
            color=np.array([0.5, 0.5, 0.5]),  # optional gray
        )

        # add robot 
        robot_prim_path = "/World/so101"
        path_to_robot_usd = ASSET_PATH + "/so101/TheRobotStudio/so101_new_calib/so101_new_calib_physx.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self.robot = Robot(robot_prim_path)

        # Add a target for the robot to move to
        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = SingleXFormPrim(
            "/World/target",
            name="target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.392173, 0, 0.223888]),
            orientation=np.array([0.7071, 0.0, 0.7071, 0.0])
        )

        

        if scenaio == "1D chess":
            BASE_POSITION, BASE_SIZE = np.array([0.4, 0, 0]), 0.04

            chess_piece_size = BASE_SIZE / 0.06  # Scale factor to make the chess pieces fit the board
            chess_pieces = [
                add_one_chess_piece("king", "white", 
                                    position=BASE_POSITION - np.array([4 * BASE_SIZE, 0, 0]), 
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "white", 
                                    position=BASE_POSITION -  np.array([3 *BASE_SIZE, 0, 0]), 
                                    orientation=np.array([0.7071, 0, 0, 0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("rook", "white", 
                                    position=BASE_POSITION - np.array([BASE_SIZE * 2, 0, 0]), 
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("rook", "black", 
                                    position=BASE_POSITION + np.array([BASE_SIZE * 1, 0.0, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "black", 
                                    position=BASE_POSITION + np.array([BASE_SIZE * 2, 0.0, 0]),
                                    orientation=np.array([0.7071, 0, 0, -0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("king", "black", 
                                    position=BASE_POSITION + np.array([BASE_SIZE * 3, 0.0, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),               
                ]
            
            chess_board = add_chessboard(
                position=BASE_POSITION + np.array([- 4 * BASE_SIZE , 0, 0.001]), 
                orientation=np.array([1, 0, 0, 0]), 
                scale=np.array([BASE_SIZE / 0.04, BASE_SIZE / 0.04, 1.0]),
                board_size="1x6"
                )
        
        elif scenaio == "3x3 chess":
            BASE_POSITION, BASE_SIZE = np.array([0.24, 0, 0]), 0.04

            chess_board = add_chessboard(
                position=BASE_POSITION + np.array([0, 0, 0.001]),  # move the chessboard slightly above the ground to avoid z-fighting
                orientation=np.array([1, 0, 0, 0]), 
                scale=np.array([BASE_SIZE / 0.06, BASE_SIZE / 0.06, 1.0]),
                board_size="3x3"
                )
            
            chess_piece_size = BASE_SIZE / 0.06  # Scale factor to make the chess pieces fit the board
            chess_pieces = [
                add_one_chess_piece("pawn", "white", piece_number=0,
                                    position=BASE_POSITION - np.array([BASE_SIZE, BASE_SIZE, 0]), 
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "white", piece_number=1,
                                    position=BASE_POSITION -  np.array([BASE_SIZE, 0, 0]), 
                                    orientation=np.array([0.7071, 0, 0, 0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "white", piece_number=2,
                                    position=BASE_POSITION - np.array([BASE_SIZE, -BASE_SIZE, 0]), 
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),

                add_one_chess_piece("pawn", "black", piece_number=0,
                                    position=BASE_POSITION + np.array([BASE_SIZE, BASE_SIZE, 0]), 
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "black", piece_number=1,
                                    position=BASE_POSITION +  np.array([BASE_SIZE, 0, 0]), 
                                    orientation=np.array([0.7071, 0, 0, 0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "black", piece_number=2,
                                    position=BASE_POSITION + np.array([BASE_SIZE, -BASE_SIZE, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
            ]

        elif scenaio == "4x4 chess":
            # Mallett, Hill & Boyer (1980) minichess setup: each side has a King, 2 Knights, a Rook, and 4 pawns.
            # https://en.wikipedia.org/wiki/Minichess
            BASE_POSITION, BASE_SIZE = np.array([0.24, 0, 0]), 0.04

            chess_board = add_chessboard(
                position=BASE_POSITION + np.array([0, 0, 0.001]),  # move the chessboard slightly above the ground to avoid z-fighting
                orientation=np.array([1, 0, 0, 0]),
                scale=np.array([BASE_SIZE / 0.06, BASE_SIZE / 0.06, 1.0]),
                board_size="4x4"
                )

            chess_piece_size = BASE_SIZE / 0.06  # Scale factor to make the chess pieces fit the board

            # ranks run along x (white back rank -> black back rank), files run along y (a -> d)
            white_back_rank, white_pawn_rank = -1.5 * BASE_SIZE, -0.5 * BASE_SIZE
            black_pawn_rank, black_back_rank = 0.5 * BASE_SIZE, 1.5 * BASE_SIZE
            file_a, file_b, file_c, file_d = -1.5 * BASE_SIZE, -0.5 * BASE_SIZE, 0.5 * BASE_SIZE, 1.5 * BASE_SIZE

            chess_pieces = [
                # white back rank: a1=King, b1/c1=Knights, d1=Rook
                add_one_chess_piece("king", "white",
                                    position=BASE_POSITION + np.array([white_back_rank, file_a, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "white", piece_number=0,
                                    position=BASE_POSITION + np.array([white_back_rank, file_b, 0]),
                                    orientation=np.array([0.7071, 0, 0, 0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "white", piece_number=1,
                                    position=BASE_POSITION + np.array([white_back_rank, file_c, 0]),
                                    orientation=np.array([0.7071, 0, 0, 0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("rook", "white",
                                    position=BASE_POSITION + np.array([white_back_rank, file_d, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),

                # white pawns
                add_one_chess_piece("pawn", "white", piece_number=0,
                                    position=BASE_POSITION + np.array([white_pawn_rank, file_a, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "white", piece_number=1,
                                    position=BASE_POSITION + np.array([white_pawn_rank, file_b, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "white", piece_number=2,
                                    position=BASE_POSITION + np.array([white_pawn_rank, file_c, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "white", piece_number=3,
                                    position=BASE_POSITION + np.array([white_pawn_rank, file_d, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),

                # black pawns
                add_one_chess_piece("pawn", "black", piece_number=0,
                                    position=BASE_POSITION + np.array([black_pawn_rank, file_a, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "black", piece_number=1,
                                    position=BASE_POSITION + np.array([black_pawn_rank, file_b, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "black", piece_number=2,
                                    position=BASE_POSITION + np.array([black_pawn_rank, file_c, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("pawn", "black", piece_number=3,
                                    position=BASE_POSITION + np.array([black_pawn_rank, file_d, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),

                # black back rank: a4=Rook, b4/c4=Knights, d4=King (180-degree rotation of the white back rank)
                add_one_chess_piece("rook", "black",
                                    position=BASE_POSITION + np.array([black_back_rank, file_a, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "black", piece_number=0,
                                    position=BASE_POSITION + np.array([black_back_rank, file_b, 0]),
                                    orientation=np.array([0.7071, 0, 0, -0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("knight", "black", piece_number=1,
                                    position=BASE_POSITION + np.array([black_back_rank, file_c, 0]),
                                    orientation=np.array([0.7071, 0, 0, -0.7071]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
                add_one_chess_piece("king", "black",
                                    position=BASE_POSITION + np.array([black_back_rank, file_d, 0]),
                                    orientation=np.array([1, 0, 0, 0]),
                                    scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                    ),
            ]

        elif scenaio == "8x8 chess":
            # Standard full chess setup.
            BASE_POSITION, BASE_SIZE = np.array([0.30, 0, 0]), 0.04

            chess_board = add_chessboard(
                position=BASE_POSITION + np.array([0, 0, 0.001]),  # move the chessboard slightly above the ground to avoid z-fighting
                orientation=np.array([1, 0, 0, 0]),
                scale=np.array([BASE_SIZE / 0.06, BASE_SIZE / 0.06, 1.0]),
                board_size="8x8"
                )

            chess_piece_size = BASE_SIZE / 0.06  # Scale factor to make the chess pieces fit the board

            files = [(i - 3.5) * BASE_SIZE for i in range(8)]  # files a -> h
            white_back_rank, white_pawn_rank = -3.5 * BASE_SIZE, -2.5 * BASE_SIZE
            black_pawn_rank, black_back_rank = 2.5 * BASE_SIZE, 3.5 * BASE_SIZE

            back_rank_order = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
            identity_orientation = np.array([1, 0, 0, 0])
            white_knight_orientation = np.array([0.7071, 0, 0, 0.7071])
            black_knight_orientation = np.array([0.7071, 0, 0, -0.7071])

            chess_pieces = []
            for color, back_rank_x, pawn_rank_x, knight_orientation in (
                ("white", white_back_rank, white_pawn_rank, white_knight_orientation),
                ("black", black_back_rank, black_pawn_rank, black_knight_orientation),
            ):
                piece_count = {}
                for file_x, piece_name in zip(files, back_rank_order):
                    piece_number = piece_count.get(piece_name, 0)
                    piece_count[piece_name] = piece_number + 1
                    orientation = knight_orientation if piece_name == "knight" else identity_orientation
                    chess_pieces.append(
                        add_one_chess_piece(piece_name, color, piece_number=piece_number,
                                            position=BASE_POSITION + np.array([back_rank_x, file_x, 0]),
                                            orientation=orientation,
                                            scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                            )
                        )
                for pawn_number, file_x in enumerate(files):
                    chess_pieces.append(
                        add_one_chess_piece("pawn", color, piece_number=pawn_number,
                                            position=BASE_POSITION + np.array([pawn_rank_x, file_x, 0]),
                                            orientation=identity_orientation,
                                            scale=np.array([chess_piece_size, chess_piece_size, chess_piece_size])
                                            )
                        )


        return self.robot, self._target, *chess_pieces
    
    
    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[2, 0.8, 1], target=[0, 0, 0], camera_prim_path="/OmniverseKit_Persp")

        # Set up robot
        # if not self.robot.handles_initialized:
        self.robot.initialize()

        # Initialize the script generator
        self._script_generator = self.my_script()

        print("[SO101ExampleScript] Robot handles initialized.")
        return
    
    def update(self, step: float):
        self.step += 1
        if self.step % 100 == 0:
            print("[SO101ExampleScript] Update called at step ", step, " with total step count ", self.step)

        # Execute one waypoint of the active motion plan every waypoint_step_interval physics steps
        if self._active_path is not None and self._path_index < len(self._active_path):
            self._waypoint_tick += 1
            if self._waypoint_tick >= self.waypoint_step_interval:
                self._waypoint_tick = 0
                q = self._active_path[self._path_index]
                current_joints = self.robot.get_joint_positions()
                gripper_status = current_joints[5:6]
                joint_positions = np.concatenate([q[:5], gripper_status])
                self.robot.apply_action(ArticulationAction(joint_positions))
                self._path_index += 1
                if self._path_index >= len(self._active_path):
                    self._active_path = None
                    print("Motion plan execution complete.")
        
    def reset(self):
        self._script_generator = self.my_script()
        return
        
    ################################################################################# 
    # Update #
    #################################################################################

    def my_script(self):
        translation_target, orientation_target = self._target.get_world_pose()
        yield ()
        
        yield from self.gripper_action(open=True)
        
        yield from self.gripper_action(open=False)


    def gripper_action(self, open=True, atol=0.01):
        """
        This is an example function for the user to implement to control the gripper.  The function can be called in the script generator to open or close the gripper as needed.  Note that how to control the gripper will depend on how the gripper is implemented in the USD, so the user will need to customize this function based on their specific gripper.
        """
        if open:
            print("Opening gripper...")
            open_gripper_action = ArticulationAction(np.array([1.0]), joint_indices=np.array([5]))
            self.robot.apply_action(open_gripper_action)
            self._gripper_target = 1.0
            return True

        else:
            print("Closing gripper...")
            close_gripper_action = ArticulationAction(np.array([0.0]), joint_indices=np.array([5]))
            self.robot.apply_action(close_gripper_action)
            self._gripper_target = 0.0
            return True
        
    def is_idle(self, gripper_atol=0.05):
        """Returns True when both the motion path and gripper have reached their targets."""
        if self._active_path is not None:
            return False
        if self._gripper_target is not None:
            current = self.robot.get_joint_positions()[5]
            if not np.isclose(current, self._gripper_target, atol=gripper_atol):
                return False
            self._gripper_target = None
        return True

    ### CuMotion IK and motion planning example ###

    def cumotion_add_pose_target(self,
        translation: np.ndarray, 
        orientation: np.ndarray, # wxyz format,
        orientation_deviation: float = 0.0,
    ):
        # Create a task space target for the IK solver.
        target_translation = translation
        target_rotation = cumotion.Rotation3(*orientation)

        # Create the translation and orientation constraints.
        translation_constraint = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
            target_translation
        )
        orientation_constraint = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
            target_rotation,
            deviation_limit=math.radians(orientation_deviation),
        )

        # orientation_constraint = cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        #     tool_frame_axis=np.array([0.0, 0.0, 1.0]),   # tool's Z axis
        #     world_target_axis=np.array([0.0, 0.0, -1.0]), # must point down in world
        #     axis_deviation_limit=math.radians(10)          # optional tolerance on axis alignment
        # )

        # Create the task-space target.
        task_space_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
            translation_constraint, orientation_constraint
        )

        # Solve again with the updated world view.
        results_with_obstacle = self.ik_solver.solve(task_space_target)

        # Check the status after updating the world.
        status_with_obstacle = results_with_obstacle.status()
        if status_with_obstacle == cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS:
            print("\nIK solve with obstacle succeeded!")
        else:
            print("\nIK solve with obstacle failed.")
            return False

        # Get the c-space positions with obstacle.
        cspace_positions_with_obstacle = results_with_obstacle.cspace_positions()
        print(f"Number of solutions found with obstacle: {len(cspace_positions_with_obstacle)}")

        if not self.robot.handles_initialized or len(cspace_positions_with_obstacle) == 0:
            return False

        q_target = cspace_positions_with_obstacle[0].astype(np.float64)
        q_current = self.robot.get_joint_positions()[:len(self.default_cspace)].astype(np.float64)

        # Plan a smooth interpolated path from current joint state to the IK target
        plan_result = self.planner.plan_to_cspace_target(q_current, q_target, True)

        if plan_result.path_found:
            self._active_path = plan_result.interpolated_path
            print(f"Motion plan found with {len(self._active_path)} waypoints.")
        else:
            # Fallback: single-step jump if planning fails
            print("Motion planning failed, applying IK solution directly.")
            self._active_path = [q_target]

        self._path_index = 0
        self._target.set_world_pose(position=translation, orientation=orientation)
        return True
    
    ####### Pick up and Place ##########
    def pick_piece(self, 
                   piece_prim_path: str,
                   orientation_deviation: float = 30.0,
                   ):
        # Get the world pose of the piece
        piece_prim = SingleXFormPrim(piece_prim_path)
        piece_translation, _ = piece_prim.get_world_pose()
        
        print("Picking piece at position: ", piece_translation)

        pick_translation, pick_orientation = get_piece_pick_up_pose(
            piece_prim_path,
            )

        # Use the cumotion_add_pose_target function to move the robot's end effector to the piece
        success = self.cumotion_add_pose_target(pick_translation, pick_orientation, orientation_deviation=orientation_deviation)
        if not success:
            carb.log_error("Failed to move to the piece.")
            return False

        # Close the gripper to grasp the piece
        # yield from self.gripper_action(open=False)

        return True
    
    def move_relative(self,
        translation_offset: np.ndarray,
        orientation_offset: np.ndarray = None, # in wxyz format
        orientation_deviation: float = 30.0,
    ):
        # Get current target pose
        current_translation, current_orientation = self._target.get_world_pose()

        # Compute new target pose by applying the relative offset
        new_translation = current_translation + translation_offset
        if orientation_offset is not None:
            new_orientation = quaternion_multiply(current_orientation, orientation_offset)
        else:
            new_orientation = current_orientation

        # Move to the new target pose
        success = self.cumotion_add_pose_target(new_translation, new_orientation, orientation_deviation=orientation_deviation)
        if not success:
            carb.log_error("Failed to move to the new position.")
            return False

        return True

    ####### Target Following ##########

    def follow_target(self,
                      position_threshold: float = 0.03,
                      orientation_threshold: float = 0.1,
                      orientation_deviation: float = 20.0
                      ):
        """Read the current world pose of self._target and move the robot end-effector there.

        Only triggers IK if the target has moved more than position_threshold (meters) or
        orientation_threshold (radians) from the current end-effector pose.
        """
        target_translation, target_orientation = self._target.get_world_pose()

        # Forward kinematics from current joint state to get current EE pose
        current_joints = self.robot.get_joint_positions()[:len(self.default_cspace)]
        current_ee_pose = self.kinematics.pose(current_joints, self.end_effector_frame)

        position_diff = np.linalg.norm(target_translation - current_ee_pose.translation)
        orientation_diff = cumotion.Rotation3.distance(
            cumotion.Rotation3(*target_orientation),
            current_ee_pose.rotation,
        )

        print(f"Target position: {target_translation}, Current EE position: {current_ee_pose.translation}, Position diff: {position_diff}, Orientation diff: {orientation_diff}")

        if position_diff > position_threshold or orientation_diff > orientation_threshold:
            self.cumotion_add_pose_target(target_translation, target_orientation, orientation_deviation=orientation_deviation)