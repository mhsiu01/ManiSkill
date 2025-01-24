from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

# Randomization-specific imports
from mani_skill.utils.structs.actor import Actor
from ...utils.randomization import visual
from mani_skill.envs.utils import randomization
from .push_cube import PushCubeEnv

@register_env("PushCubeRandomized-v1", max_episode_steps=50)
class PushCubeRandomizedEnv(PushCubeEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # Vanilla PushCube init
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    @property
    def _default_sensor_configs(self):
        p_perturb = 0.05
        q_perturb = np.pi / 24
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * p_perturb - 0.5*p_perturb,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-q_perturb, q_perturb)
            ),
        )
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]
    
    def _load_scene(self, options: dict):
        self.table_scenes = []
        self.tables = []
        self.grounds = []
        self.objs = []
        self.goal_regions = []

        # Load sub-scenes one by one
        for i in range(self.num_envs):
            # we use a prebuilt scene builder class that automatically loads in a floor and table.
            table_scene = TableSceneBuilder(
                env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
            )
            table_scene.build(scene_idxs=[i])
            self.tables.append(table_scene.table)
            self.grounds.append(table_scene.ground)
            self.table_scenes.append(table_scene) # Not for merging, but for initialization
    
            # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
            # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
            obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=np.array([12, 42, 160, 255]) / 255,
                name=f"cube-{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
                scene_idxs=[i],
            )                
            self.objs.append(obj)
    
            # we also add in red/white target to visualize where we want the cube to be pushed to
            # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
            # we finally specify the body_type to be "kinematic" so that the object stays in place
            goal_region = actors.build_red_white_target(
                self.scene,
                radius=self.goal_radius,
                thickness=1e-5,
                name=f"goal_region-{i}",
                add_collision=False,
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
                scene_idxs=[i],
            )
            self.goal_regions.append(goal_region)
    
            # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
            # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
            # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
            # and are there just for generating evaluation videos.
            # self._hidden_objects.append(self.goal_region)
        print("Actors spawned.")

        # Randomize texture, lighting, color.
        visual.randomize_env(self, options)
                            
        # Merge actors across all sub-scenes, except for table_scenes, which are not Actors.
        # self.table_scene = Actor.merge(self.table_scenes, name="table-workspace")
        self.table = Actor.merge(self.tables, name="table")
        self.ground = Actor.merge(self.grounds, name="ground")
        self.obj = Actor.merge(self.objs, name="cube")
        self.goal_region = Actor.merge(self.goal_regions, name="goal_region")

        # Mount camera for later pose randomization
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")


    def _load_lighting(self, options: dict):
        if "lighting" not in options.keys():
            super()._load_lighting(options)
        else:
            visual.load_custom_lighting(env=self, options=options)

    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            for i,table_scene in enumerate(self.table_scenes):
                # The robot is a batch-level object getting roped into individual sub-scene initializations
                # during table_scene init
                table_scene.initialize(env_idx, scene_idxs=[i])

            # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
            # and we further rotate 90 degrees on the y-axis to make the target object face up
            target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
