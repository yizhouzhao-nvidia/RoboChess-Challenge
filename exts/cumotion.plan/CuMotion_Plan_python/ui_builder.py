# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio

import numpy as np
import time

import omni.timeline
import omni.ui as ui
import carb

from isaacsim.core.api.world import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import create_new_stage, get_current_stage
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import CollapsableFrame, StateButton
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType
from pxr import Sdf, UsdLux

import omni.replicator.core as rep

from .scenario import FrankaRmpFlowExampleScript
from .cumotion.piper import PiperCuMotionPlanExample
from .cumotion.so101 import SO101ExampleScript

class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        self._auto_sequence_gen = None   # active full-sequence generator
        self._seq_step_triggered = False  # True while waiting for current step to finish

        # Run initialization for the provided example
        self._on_init()

        # replicator
        self.rp = None
        self.cosmos_writer = None

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

            self._so101_scenario_state_btn.reset()
            self._so101_scenario_state_btn.enabled = False


    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        if self._auto_sequence_gen is None:
            return

        scenario = self._so101_scenario
        robot_idle = scenario.is_idle()

        if not self._seq_step_triggered:
            try:
                next(self._auto_sequence_gen)
                self._seq_step_triggered = True
            except StopIteration:
                print("Full sequence complete.")
                self._auto_sequence_gen = None
                self._seq_step_triggered = False
        elif robot_idle:
            # Both motion path and gripper have settled; allow next step next tick
            self._seq_step_triggered = False

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)

        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

        run_scenario_frame = CollapsableFrame("Run Scenario")

        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

        so101_controls_frame = CollapsableFrame("Lerobot SO101 Controls", collapsed=False)

        with so101_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._so101_load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_so101_scene, setup_post_load_fn=self._setup_so101_scenario
                )
                self._so101_load_btn.set_world_settings(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)
                self.wrapped_ui_elements.append(self._so101_load_btn)

                self._so101_reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_so101_reset_btn
                )
                self._so101_reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._so101_reset_btn)

                ui.Spacer(height=10)

                self._so101_scenario_state_btn = StateButton(
                    "Run SO101Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._so101_update_scenario,
                )
                self._so101_scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._so101_scenario_state_btn)

                ui.Spacer(height=10)
                ui.Label("Cumotion Tests", height = 20)
                ui.Line()
                ui.Button("CuMotion Init", width=150, clicked_fn=self.cumotion_init)
                ui.Button("CuMotion Target", width=150, clicked_fn=self.cumotion_target)
                ui.Button("Open Gripper", width=150, clicked_fn=self.open_gripper)
                ui.Button("Close Gripper", width=150, clicked_fn=self.close_gripper)
                ui.Button("Test Pick Piece", width=200, clicked_fn=self.pick_piece)
                ui.Button("Run Full Sequence", width=200, clicked_fn=self.run_full_sequence)
                ui.Button("Stop Sequence", width=200, clicked_fn=self.stop_sequence)

                

                ui.Line()
                ui.Button("Debug", height = 40, clicked_fn=self.debug)
                ui.Button("Debug2", height = 40, clicked_fn=self.debug2)
                


    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Deleted/Replaced
    ######################################################################################

    def _on_init(self):
        self._articulation = None
        self._cuboid = None
        self._scenario = FrankaRmpFlowExampleScript()

        # cumotion example
        self.cumotion_example = PiperCuMotionPlanExample()

        # so101 example
        self._so101_scenario = SO101ExampleScript()

    def _add_light_to_stage(self):
        """
        A new stage does not have a light by default.  This function creates a spherical light
        """
        sphereLight = UsdLux.SphereLight.Define(get_current_stage(), Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        SingleXFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._add_light_to_stage()

        loaded_objects = self._scenario.load_example_assets()

        # Add user-loaded objects to the World
        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by their setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._scenario.setup()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._scenario.reset()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float, *_):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        This function will repeatedly advance the script in scenario.py until it is finished.

        Args:
            step (float): The dt of the current physics step
        """
        done = self._scenario.update(step)
        if done:
            self._scenario_state_btn.enabled = False

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False

        self._so101_scenario_state_btn.reset()
        self._so101_scenario_state_btn.enabled = False
        self._so101_reset_btn.enabled = False



    #################################################################################
    ############### so101_controls_frame ############################################
    #################################################################################

    def _setup_so101_scene(self):
        create_new_stage()
        self._add_light_to_stage()

        loaded_objects = self._so101_scenario.load_example_assets()

        # Add user-loaded objects to the World
        world = World.instance()
        for loaded_object in loaded_objects:
            world.scene.add(loaded_object)

    def _setup_so101_scenario(self):
        print("Post Setting up SO101 scenario")
        
        self._so101_scenario.setup()

        
        # TODO: # UI management
        self._so101_scenario_state_btn.reset()
        self._so101_scenario_state_btn.enabled = True
        self._so101_reset_btn.enabled = True

    def _so101_update_scenario(self, step: float, *_):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        This function will repeatedly advance the script in scenario.py until it is finished.

        Args:
            step (float): The dt of the current physics step
        """
        done = self._so101_scenario.update(step)
        if done:
            self._so101_scenario_state_btn.enabled = False
        
        async def _capture_async():
            try:
                await rep.orchestrator.step_async(
                    delta_time=0.0,
                    rt_subframes=2,
                    pause_timeline=False,
                )
            except Exception as e:
                print(f"Error during rep.orchestrator.step_async(): {e}")

                
        if self.rp and self.cosmos_writer:
            # rep.orchestrator.step(pause_timeline=False)
            omni.kit.async_engine.run_coroutine(_capture_async())
            print("rep.orchestrator.step() called in _so101_update_scenario()")

    def _on_so101_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the position it was in when it was created in self._setup_scene().
        """
        self._so101_scenario.reset()

        # UI management
        self._so101_scenario_state_btn.reset()
        self._so101_scenario_state_btn.enabled = True

    def open_gripper(self):
        if self._so101_scenario.robot is not None and self._so101_scenario.robot.handles_initialized:
            self._so101_scenario.gripper_action(open=True)

    def close_gripper(self):
        if self._so101_scenario.robot is not None and self._so101_scenario.robot.handles_initialized:
            self._so101_scenario.gripper_action(open=False)

    def cumotion_init(self):
        print("cumotion_init function called")
        self._so101_scenario.cumotion_setup()

    def cumotion_target(self):
        print("cumotion_target function called")
        self._so101_scenario.cumotion_add_pose_target(
            translation=np.array([0.35, 0, 0.2]),
            orientation=np.array([0.7071, 0, 0.7071, 0])  # wxyz format
        )

    def _pick_piece_steps(self):

        print("Step 1: open gripper")
        self._so101_scenario.gripper_action(open=True)
        yield

        print("Step 2: move to piece")
        self._so101_scenario.pick_piece("/World/white_rook")
        yield

        print("Step 3: close gripper")
        self._so101_scenario.gripper_action(open=False)
        yield

        print("Step 4: move up")
        self._so101_scenario.move_relative(
            translation_offset=np.array([0.0, 0.0, 0.06]),
        )
        yield

        print("Step 5: move to target")
        self._so101_scenario.move_relative(
            translation_offset=np.array([0.04, 0.0, 0]),
        )
        yield

        print("Step 5.5: move to target")
        self._so101_scenario.move_relative(
            translation_offset=np.array([0.0, 0.0, -0.06]),
        )
        yield

        print("Step 6: open gripper")
        self._so101_scenario.gripper_action(open=True)
        yield

        print("Step 7: move up")
        self._so101_scenario.move_relative(
            translation_offset=np.array([0.0, 0.0, 0.06]),
        )
        yield


        # black rook rotation
        # [0.15305, 0.69035, 0.69035, 0.15305]

    def pick_piece(self):
        if not hasattr(self, "_pick_piece_gen") or self._pick_piece_gen is None:
            self._pick_piece_gen = self._pick_piece_steps()
        try:
            next(self._pick_piece_gen)
        except StopIteration:
            print("pick_piece sequence complete, resetting")
            self._pick_piece_gen = None

    def run_full_sequence(self):
        if self._auto_sequence_gen is not None:
            print("Sequence already running; click Stop Sequence first.")
            return
        print("Starting full pick-and-place sequence.")
        self._auto_sequence_gen = self._pick_piece_steps()
        self._seq_step_triggered = False

    def stop_sequence(self):
        self._auto_sequence_gen = None
        self._seq_step_triggered = False
        print("Sequence stopped.")

    
    def _on_post_so101_reset_btn(self):
        """
        Performs any necessary cleanup after the Reset Button is pressed.
        """
        async def stop_replicator():
            if self.rp and self.cosmos_writer:
                await rep.orchestrator.wait_until_complete_async()
                self.cosmos_writer.detach()
                self.rp.destroy()
                self.rp = None
                self.cosmos_writer = None
                print("Replicator stopped and resources cleaned up.")

        asyncio.ensure_future(stop_replicator())

    
    def debug(self):
        print("Debug")
        from isaacsim.core.prims import SingleXFormPrim
        robot_prim = SingleXFormPrim("/World/so101")
        base_t, base_r = robot_prim.get_world_pose()
        print("Robot base world pose:", base_t, base_r)
        # print("cuMotion default tool pose:", self.default_tool_pose.translation)
        self._so101_scenario.follow_target(
            orientation_deviation=30,  # degrees
        )

    def debug2(self):
        print("Debug2")
        # from .cumotion.chess_utils import get_bounding_box
        # min_pt, max_pt, size, center = get_bounding_box("/World/knight")
        # print("Bounding box - min:", min_pt, "max:", max_pt, "size:", size, "center:", center)

        
        # Disable capture on play on the new stage, data is captured manually using the step function
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # camera
        stage = omni.usd.get_context().get_stage()
        CAMERA_PATH = "/OmniverseKit_Persp"
        camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
        if not camera_prim.IsValid():
            print(f"Camera prim not found at path: {CAMERA_PATH}, exiting")
            return

        # tickRate=0 forces autotrigger so the sensor cameras stay in sync with rep.orchestrator.step
        # under multi-tick rendering.
        if camera_prim.HasAttribute("omni:sensor:tickRate"):
            camera_prim.GetAttribute("omni:sensor:tickRate").Set(0.0)


        """Run the synthetic data generation pipeline and capture video clips."""
        self.rp = rep.create.render_product(CAMERA_PATH, (1280, 720))
        self.cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
        backend = rep.backends.get("DiskBackend")

        from .cumotion.chess_utils import DATASET_PATH
        import os
        out_dir = os.path.join(DATASET_PATH, f"rep_test_{int(time.time())}")
        print(f"output_directory: {out_dir}")
        backend.initialize(output_dir=out_dir)
        self.cosmos_writer.initialize(
            backend=backend, use_instance_id=False, segmentation_mapping=None
        )
        self.cosmos_writer.attach(self.rp)
