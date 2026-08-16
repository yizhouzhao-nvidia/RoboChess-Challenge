"""Measure everything a new arm needs before it can be added to the chess task.

Adding a robot to :mod:`robochess.tasks.manager_based.chess.franka_chess_env_cfg`
requires facts that cannot be guessed from a USD path: which body the IK should
drive, where that body's prim actually lives, how far the TCP sits from it, which
axis the fingers close along, and how wide they open. Every one of those has bitten
this project at least once, so this script measures them instead.

.. code-block:: bash

    python lab/scripts/probe_robot.py --robot rebot --headless
    python lab/scripts/probe_robot.py --robot all --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe an arm for the chess picking task.")
parser.add_argument("--robot", type=str, default="all", help="Robot key from CHESS_ROBOTS, or 'all'.")
parser.add_argument("--settle_steps", type=int, default=90, help="Steps to hold each gripper command.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.assets import Articulation
from pxr import Usd, UsdGeom, UsdPhysics

from robochess.tasks.manager_based.chess.robot_configs import CHESS_ROBOTS


def rigid_body_prim_paths(root: str) -> list[str]:
    """Every rigid-body prim under ``root``, in traversal order."""
    stage = omni.usd.get_context().get_stage()
    return [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(root) and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def geometry_extents(root: str, reference: str, collision_only: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounding box of the geometry under ``root``, expressed in ``reference``'s frame.

    Traverses instance proxies: several of these assets reference their meshes from a
    shared ``instances.usda``, and a plain ``Stage.Traverse()`` walks straight past
    them, reporting an arm with no geometry at all.
    """
    stage = omni.usd.get_context().get_stage()
    reference_prim = stage.GetPrimAtPath(reference)
    if not reference_prim.IsValid():
        return None
    inverse = np.linalg.inv(np.array(UsdGeom.Xformable(reference_prim).ComputeLocalToWorldTransform(0)).T)
    lower, upper, found = np.full(3, 1e9), np.full(3, -1e9), 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        if collision_only and not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        elif prim.IsA(UsdGeom.Cube):
            half = (UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
            points = np.array([[x * half, y * half, z * half] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        else:
            continue
        transform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
        points = (inverse[:3, :3] @ points.T).T + inverse[:3, 3]
        lower, upper, found = np.minimum(lower, points.min(0)), np.maximum(upper, points.max(0)), found + 1
    return (lower, upper) if found else None


def finger_offsets(robot: Articulation, ee_body: str, finger_bodies: list[str]) -> dict[str, np.ndarray]:
    """Each finger body's origin expressed in the end-effector body frame."""
    ee_index = robot.find_bodies(ee_body)[0][0]
    ee_pos = robot.data.body_pos_w.torch[:, ee_index]
    ee_quat = robot.data.body_quat_w.torch[:, ee_index]
    offsets = {}
    for name in finger_bodies:
        matches = robot.find_bodies(name)[0]
        if not matches:
            print(f"    [warn] no body named '{name}'; known bodies: {robot.body_names}")
            continue
        index = matches[0]
        pos_b, _ = math_utils.subtract_frame_transforms(
            ee_pos, ee_quat, robot.data.body_pos_w.torch[:, index], robot.data.body_quat_w.torch[:, index]
        )
        offsets[name] = pos_b[0].cpu().numpy()
    return offsets


def dominant_axis(vector: np.ndarray) -> tuple[np.ndarray, str]:
    """Snap a vector to the nearest signed principal axis, with a readable label."""
    index = int(np.argmax(np.abs(vector)))
    sign = 1.0 if vector[index] >= 0 else -1.0
    axis = np.zeros(3)
    axis[index] = sign
    return axis, f"{'+-'[sign < 0]}{'XYZ'[index]}"


def infer_axes(open_offsets: dict[str, np.ndarray], closed_offsets: dict[str, np.ndarray], spec) -> None:
    """Print ChessRobotSpec values inferred from the finger poses.

    Everything here comes from the physics view (``robot.data.body_pos_w``), which is
    the only trustworthy source once the arm has moved -- USD xform values are the
    *authored* pose and go stale the moment the simulation steps.
    """
    names = list(open_offsets)
    travel = open_offsets[names[0]] - closed_offsets[names[0]]
    closing, closing_label = dominant_axis(travel)

    # The object ends up midway between the fingers, so that midpoint is both the
    # approach direction and the first estimate of the TCP.
    midpoint = np.mean([closed_offsets[name] for name in names], axis=0)
    approach, approach_label = dominant_axis(midpoint)
    separation = float(np.linalg.norm(open_offsets[names[0]] - open_offsets[names[1]]))

    print("\nmeasured ChessRobotSpec values (physics, not USD):")
    print(f"    approach_axis={tuple(approach)}      # {approach_label}, finger midpoint {np.round(midpoint, 4)}")
    print(f"    closing_axis={tuple(closing)}       # {closing_label}, travel {np.linalg.norm(travel) * 1000:.1f} mm/finger")
    print(f"    tcp_offset={tuple(np.round(midpoint, 4))}   # refine along the approach to sit at the finger pads")
    print(f"    max_opening={separation:.4f}          # {separation * 1000:.1f} mm between finger origins")
    print(f"  current spec: approach={spec.approach_axis} closing={spec.closing_axis} "
          f"tcp={spec.tcp_offset} max_opening={spec.max_opening}")


def probe(key: str) -> None:
    spec = CHESS_ROBOTS[key]
    print(f"\n{'=' * 78}\n=== {key}: {spec.articulation.spawn.usd_path}\n{'=' * 78}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    root = "/World/Robot"
    robot = Articulation(spec.articulation.replace(prim_path=root))
    sim.reset()

    print(f"\njoints ({robot.num_joints}):")
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    for name, (low, high) in zip(robot.joint_names, limits):
        print(f"    {name:<24s} [{low:8.4f}, {high:8.4f}]")
    print(f"\nbodies ({robot.num_bodies}): {robot.body_names}")

    print("\nrigid-body prim paths (nesting matters for FrameTransformerCfg):")
    for path in rigid_body_prim_paths(root):
        print(f"    {path}")

    finger_ids, finger_names = robot.find_joints(spec.gripper_joints)
    arm_ids, arm_names = robot.find_joints(spec.arm_joints)
    print(f"\nresolved arm joints    : {arm_names}")
    print(f"resolved gripper joints: {finger_names}")

    # Drive the gripper to each end of its stroke and watch where the fingers go.
    open_offsets: dict[str, np.ndarray] = {}
    closed_offsets: dict[str, np.ndarray] = {}
    for label, command, store in (
        ("OPEN", spec.open_command, open_offsets),
        ("CLOSE", spec.close_command, closed_offsets),
    ):
        targets = robot.data.joint_pos.torch.clone()
        for joint_expr, value in command.items():
            ids, _ = robot.find_joints(joint_expr)
            targets[:, ids] = value
        for _ in range(args_cli.settle_steps):
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
            sim.step()
            robot.update(1.0 / 120.0)
        gap = robot.data.joint_pos.torch[0, finger_ids].cpu().numpy()
        want = targets[0, finger_ids].cpu().numpy()
        print(f"\n{label}: gripper joints = {np.round(gap, 5)}  commanded {np.round(want, 5)}"
              f"  (sum {abs(gap).sum() * 1000:.1f} mm)")
        if spec.finger_bodies:
            store.update(finger_offsets(robot, spec.ee_body, spec.finger_bodies))
            for name, offset in store.items():
                print(f"    {name:<20s} in {spec.ee_body} frame: {np.round(offset, 5)}")

    if len(open_offsets) >= 2 and len(closed_offsets) >= 2:
        infer_axes(open_offsets, closed_offsets, spec)

    # Where the collision geometry actually sits in the ee frame is what reveals the
    # tool axis: the fingers extend along it, and their far end is the TCP.
    ee_path = spec.ee_prim_path(root)
    print(f"\ncollision extents in the {spec.ee_body} frame (ee_prim_path={ee_path}):")
    # Finger prims are children of the wrist on some assets and siblings on others,
    # so resolve each by name against the rigid-body list rather than assuming.
    by_leaf = {path.rsplit("/", 1)[-1]: path for path in rigid_body_prim_paths(root)}
    subtrees = {"ee subtree": ee_path}
    for finger in spec.finger_bodies:
        if finger in by_leaf:
            subtrees[finger] = by_leaf[finger]
    for label, subtree in subtrees.items():
        for kind, collision_only in (("collision", True), ("visual+coll", False)):
            extents = geometry_extents(subtree, ee_path, collision_only)
            if extents is None:
                print(f"    {label:<16s} {kind:<12s} none")
                continue
            lower, upper = extents
            print(f"    {label:<16s} {kind:<12s} min={np.round(lower, 4)}  max={np.round(upper, 4)}")

    sim.clear_instance()


def main():
    keys = list(CHESS_ROBOTS) if args_cli.robot == "all" else [args_cli.robot]
    for key in keys:
        if key not in CHESS_ROBOTS:
            raise SystemExit(f"Unknown robot '{key}'. Choose from {list(CHESS_ROBOTS)}.")
        probe(key)


if __name__ == "__main__":
    main()
    simulation_app.close()
