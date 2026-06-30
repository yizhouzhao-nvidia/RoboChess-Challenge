# utils
import os
import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade
import omni.usd

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.api.robots.robot import Robot


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.abspath(os.path.join(FILE_PATH, "../../../../assets"))
PACKAGE_PATH = os.path.abspath(os.path.join(FILE_PATH, "../../../../package"))
DATASET_PATH = os.path.abspath(os.path.join(FILE_PATH, "../../../../dataset"))

import numpy as np
import random

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


def add_chessboard(
        position = np.array([0, 0, 0]),
        orientation = np.array([1.0, 0, 0, 0]),
        scale = np.array([1.0, 1.0, 1.0]),
        board_size: str = "1x6",
    ):
    chessboard_prim_path = "/World/Chessboard"
    chessboard_usd_path = ASSET_PATH + f"/chess/board/board_{board_size}.usdc"
    add_reference_to_stage(chessboard_usd_path, chessboard_prim_path)

    chessboard_xform = SingleXFormPrim(
        chessboard_prim_path,
        name="chessboard",
        scale=scale,
        position=position,
        orientation=orientation,
    )

    return chessboard_xform


def add_one_chess_piece(
        piece_name = "knight", 
        color = "black",
        position = np.array([0.3, 0, 0]),
        orientation = np.array([1.0, 0, 0, 0]),
        scale = np.array([1.0, 1.0, 1.0]),
        piece_number = 0,
    ):
    """
    This function is meant for the user to add any additional assets they want in the scene that are not part of the robot or target.  This function is called in ui_builder._setup_scenario() after load_example_assets() and before setup().
    """
    chess_piece_folder = ASSET_PATH + "/chess"
    
    # add a piece to the scene
    piece_prim_path = f"/World/{color}_{piece_name}_{piece_number}"
    piece_usd_path = chess_piece_folder + f"/{piece_name}.usdc"
    add_reference_to_stage(piece_usd_path, piece_prim_path)

    piece_xform = SingleXFormPrim(
        piece_prim_path,
        name=f"{color}_{piece_name}_{piece_number}",
        scale=scale,
        position=position,
        orientation=orientation,
        )
    
    # Create a material
    piece_material_path = f"/World/Looks/{color}_piece_material"
    if not omni.usd.get_context().get_stage().GetPrimAtPath(piece_material_path):
        print(f"Creating material for {color} {piece_name} at {piece_material_path}")

        piece_material = PreviewSurface(
            prim_path=piece_material_path,
            name=f"{color}_{piece_name}_material",
            color=np.array([0.02, 0.02, 0.02]) if color == "black" else np.array([0.9, 0.9, 0.9]),  # near-black or near-white
            roughness=0.4,
            metallic=0.0,
        )
    else:
        print(f"Material for {color} {piece_name} already exists at {piece_material_path}. Skipping creation.")
        piece_material = PreviewSurface(
            prim_path=piece_material_path,
            name=f"{color}_{piece_name}_material",
        )

    # Wrap the prim and bind the material
    piece_prim = SingleGeometryPrim(prim_path=piece_prim_path, name=f"{color}_{piece_name}")
    piece_prim.apply_visual_material(piece_material)


    # add rigid body and colliders to the piece
    add_rigid_body_and_colliders(piece_prim_path)

    # add physical material and bind it to the piece
    add_physical_material(
        prim_path=piece_prim_path,
        physics_material_path=f"/World/Physics_Materials/my_physics_material",
        static_friction=1.5,
        dynamic_friction=1.5,
    )

    return piece_xform


def add_rigid_body_and_colliders(
    prim_path = "/World/MyObject"
):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    # 1. Apply Rigid Body to the root prim
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)

    # PhysX extension (damping, gravity, etc.)
    physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb_api.GetDisableGravityAttr().Set(False)

    # 2. Apply Colliders to geometry descendants
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        if desc_prim.IsA(UsdGeom.Mesh):
            mesh_col_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            mesh_col_api.CreateApproximationAttr().Set("convexHull") #convexDecomposition # 


def add_physical_material(
    prim_path = "/World/MyObject",
    physics_material_path = "/World/Looks/MyPhysicsMaterial",
    static_friction = 0.5,
    dynamic_friction = 0.3,
    restitution = 0.0,
):
    stage = omni.usd.get_context().get_stage()

    # if material already exists, not need to create it again
    if stage.GetPrimAtPath(physics_material_path):
        print(f"Physics material already exists at {physics_material_path}. Skipping creation.")
        mat = UsdShade.Material(stage.GetPrimAtPath(physics_material_path))
    else:
        mat = UsdShade.Material.Define(stage, physics_material_path)

        # USD standard friction/restitution
        physics_mat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics_mat.CreateStaticFrictionAttr().Set(static_friction)
        physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
        physics_mat.CreateRestitutionAttr().Set(restitution)

        # PhysX combine modes: "average", "min", "multiply", "max"
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physx_mat.CreateFrictionCombineModeAttr().Set("max")
        physx_mat.CreateRestitutionCombineModeAttr().Set("max")

    # Bind to your prim
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def get_bounding_box(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    # Create a cache (reuse it across multiple prims for performance)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_]
    )

    # World-space bounding box
    world_bound = bbox_cache.ComputeWorldBound(prim)
    bbox_range = world_bound.GetRange()

    min_pt = bbox_range.GetMin()   # Gf.Vec3d (x, y, z)
    max_pt = bbox_range.GetMax()   # Gf.Vec3d (x, y, z)
    size   = bbox_range.GetSize()  # Gf.Vec3d (width, depth, height)
    center = bbox_range.GetMidpoint()

    return np.array(min_pt), np.array(max_pt), np.array(size), np.array(center)

def get_piece_pick_up_pose(
    prim_path = "/World/knight",
    pick_translation_offset = np.array([0.0, 0.02, 0.03]),
    pick_orientation = np.array([0, 0.7071, 0.7071, 0]),
    robot: Robot = None,
):
    """
    Function to get the pick-up pose for a chess piece. This is meant to be called in ui_builder.py when the user clicks the "Pick Piece" button, and the returned pose will be used as the target pose for cuMotion IK planning. The default implementation here just returns a hardcoded pose, but users can modify this function to implement their own logic for computing the pick-up pose based on the current scene state (e.g. by querying the piece's current position and orientation and adding some offset to compute a grasping pose).
    """

    # get piece transform
    piece_prim = SingleXFormPrim(prim_path)
    piece_translation, piece_rotation = piece_prim.get_world_pose()

    ## TODO: get robot base transform (if needed for computing the pick pose in the robot's frame)
    # if robot is not None:
    #     robot_base_translation, robot_base_rotation = robot.get_world_pose()


    # For demonstration, we will just return a pose that is slightly above the piece's current position, with a fixed orientation.
    pick_translation = piece_translation + pick_translation_offset

    # # if the chess is knight, we have to adjust the pick translation and orientation a bit to make sure the gripper doesn't collide with the knight's head
    # if "knight" in prim_path.lower():
    #     rotation_add = np.array([0.7071, 0, 0, 0.7071])  # 90 degrees rotation around Z-axis
    #     # apply the z rotation to the original pick orientation
    #     pick_orientation = quaternion_multiply(pick_orientation, rotation_add)

    return pick_translation, pick_orientation