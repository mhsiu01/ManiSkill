from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


@register_env("LiftPegUpright-v1", max_episode_steps=50)
class LiftPegUprightEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        return {
            "success": is_peg_upright & close_to_table,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        vec = torch.tensor([1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.peg)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward




# import os
# import random
# import sapien
# from mani_skill.utils.structs.actor import Actor
from ...utils.randomization import common as rand_funcs
from rand_funcs import *

@register_env("LiftPegUprightRandomized-v1", max_episode_steps=50)
class LiftPegUprightRandomizedEnv(LiftPegUprightEnv):
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.texture_files = _load_textures()
        # Vanilla init
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        
    # def _randomize_texture(self, obj):
    #     path = random.sample(self.texture_files,1)[0]
    #     texture = sapien.render.RenderTexture2D(filename=path)
    #     for part in obj._objs:
    #         for render_shape in part.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
    #             for triangle in render_shape.parts:
    #                 triangle.material.set_base_color_texture(texture)
    
    # def _randomize_color(self, obj, color):
    #     if len(color)==2:
    #         color_min = np.array(color[0])
    #         color_max = np.array(color[1])
    #         # Uniform sample within interval [color_min, color_max]
    #         final_color = color_min + np.random.rand(3)*(color_max - color_min)
    #     else:
    #         final_color = color
    #     final_color = np.append(final_color, 1.0) # Append 4th alpha channel
            
    #     for part in obj._objs:
    #         for render_shape in part.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
    #             for triangle in render_shape.parts:
    #                 triangle.material.set_base_color(final_color)

    def _load_lighting(self, options: dict):
        if "lighting" not in options.keys():
            super()._load_lighting(options)
        else:  
            common._load_custom_lighting(env=self, options=options)
    # def _load_lighting(self, options: dict):
    #     if "lighting" not in options.keys():
    #         super()._load_lighting(options)
    #     else:
    #         # Set shared shadow 
    #         shadow = self.enable_shadow
    #         # Randomized ambient light per sub-scene
    #         if "ambient" in options["lighting"]:
    #             ambient_min = np.array(options["lighting"]["ambient"][0])
    #             ambient_max = np.array(options["lighting"]["ambient"][1])
    #             for i in range(self.num_envs):
    #                 ambient_light = ambient_min + np.random.rand(3)*(ambient_max - ambient_min)
    #                 ambient_light = np.append(ambient_light, i)
    #                 self.scene.set_ambient_light(ambient_light)
    #             print("Ambient light randomized.")

    #         # Randomized directional lights per sub-scene
    #         if "directional" in options["lighting"]:
    #             for i in range(self.num_envs):
    #                 # Add config-specified directional lights per sub-scene
    #                 for d in options["lighting"]["directional"]:
    #                     direction_min = np.array(d["direction"][0])
    #                     direction_max = np.array(d["direction"][1])
    #                     direction = direction_min + np.random.rand(3)*(direction_max - direction_min)
    #                     color_min = np.array(d["color"][0])
    #                     color_max = np.array(d["color"][1])
    #                     color = color_min + np.random.rand(3)*(color_max - color_min)
    #                     self.scene.add_directional_light(
    #                         direction, color, shadow=shadow, shadow_scale=5, shadow_map_size=2048,
    #                         scene_idxs=[i],
    #                     )
    #             print("Directional lights randomized.")
    #                 # TODO: Add specified spot and point lights

    # # Convenience method forrandomizing tabletop tasks
    # def _load_table_scenes(self):
    #     self.tables = []
    #     self.grounds = []
    #     self.table_scenes = []

    #     for i in range(self.num_envs):
    #         table_scene = TableSceneBuilder(
    #             env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
    #         )
    #         table_scene.build(scene_idxs=[i])
    #         self.tables.append(table_scene.table)
    #         self.grounds.append(table_scene.ground)
    #         self.table_scenes.append(table_scene) # Not for merging, but for initialization

    #     # Merge table-related actors
    #     self.table = Actor.merge(self.tables, name="table")
    #     self.ground = Actor.merge(self.grounds, name="ground")


    def _load_scene(self, options: dict):
        common._load_table_scenes(env=self)
        
        self.pegs = []        
        for i in range(self.num_envs):
            # the peg that we want to manipulate
            peg = actors.build_twocolor_peg(
                self.scene,
                length=self.peg_half_length,
                width=self.peg_half_width,
                color_1=np.array([176, 14, 14, 255]) / 255,
                color_2=np.array([12, 42, 160, 255]) / 255,
                name=f"peg-{i}",
                body_type="dynamic",
                scene_idxs=[i],
            )
            self.pegs.append(peg)
        self.peg = Actor.merge(self.pegs, name="peg")
        print("Actors spawned.")
        
        # Check if config exists and perform randomization
        if "actors" in options:
            # If so, loop over sub-scenes and randomize
            for i in range(self.num_envs):
                # Iterate through specified actors of i-th sub-scene
                for actor_name, rand_dict in options["actors"].items():
                    actor = self.scene.actors[f"{actor_name}-{i}"] # eg. actor_name="cube", i=5 --> actor=cube 5
                    # Apply all randomizations to this actor
                    for rand_type, rand_value in rand_dict.items():
                        if rand_type=="texture" and rand_value is True:
                            common._randomize_texture(env=self, obj=actor)
                        elif rand_type=="color":
                            common._randomize_color(obj=actor, color=rand_value)
            print("Actors randomized.")


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            for i,table_scene in enumerate(self.table_scenes):
                # The robot is a batch-level object getting roped into individual sub-scene initializations
                # during table_scene init
                table_scene.initialize(env_idx, scene_idxs=[i])
            # self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

 
    