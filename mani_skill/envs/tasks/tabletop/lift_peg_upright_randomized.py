from typing import Any, Dict, Union

import numpy as np
import sapien
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


from mani_skill.utils.structs.actor import Actor
from ...utils.randomization import visual
from mani_skill.envs.utils import randomization
from .lift_peg_upright import LiftPegUprightEnv

@register_env("LiftPegUprightRandomized-v1", max_episode_steps=50)
class LiftPegUprightRandomizedEnv(LiftPegUprightEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # Vanilla init
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        
    def _load_lighting(self, options: dict):
        if "lighting" not in options.keys():
            super()._load_lighting(options)
        else:  
            visual.load_custom_lighting(env=self, options=options)

    def _load_scene(self, options: dict):
        visual.load_table_scenes(env=self)
        
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
                initial_pose=sapien.Pose(p=[0, 0, 0.1]),
                scene_idxs=[i],
            )
            self.pegs.append(peg)
        self.peg = Actor.merge(self.pegs, name="peg")
        print("Actors spawned.")
        visual.randomize_env(self, options)


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
    