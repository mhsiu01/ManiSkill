import os
from typing import Sequence, Union
import random

import numpy as np
import torch

import sapien
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.scene_builder.table import TableSceneBuilder


def load_textures():
    # Prepare list of textures to sample from.
    texture_dir = "/fast-vol/robot-colosseum/colosseum/assets/textures"
    # Prepare textures
    texture_files = os.listdir(texture_dir)
    texture_files = [os.path.join(texture_dir,f) for f in texture_files if f.endswith(".png")]
    print(f"Found {len(texture_files)} texture files.")
    return texture_files
    
def randomize_texture(env, obj):
    path = random.sample(env.texture_files,1)[0]
    texture = sapien.render.RenderTexture2D(filename=path)
    for part in obj._objs:
        for render_shape in part.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
            for triangle in render_shape.parts:
                triangle.material.set_base_color_texture(texture)

def randomize_color(obj, color):
    if len(color)==2:
        color_min = np.array(color[0])
        color_max = np.array(color[1])
        # Uniform sample within interval [color_min, color_max]
        final_color = color_min + np.random.rand(3)*(color_max - color_min)
    else:
        final_color = color
    final_color = np.append(final_color, 1.0) # Append 4th alpha channel
        
    for part in obj._objs:
        for render_shape in part.find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
            for triangle in render_shape.parts:
                triangle.material.set_base_color(final_color)


def load_custom_lighting(env, options: dict):
    # Set shared shadow 
    shadow = env.enable_shadow
    # Randomized ambient light per sub-scene
    if "ambient" in options["lighting"]:
        ambient_min = np.array(options["lighting"]["ambient"][0])
        ambient_max = np.array(options["lighting"]["ambient"][1])
        for i in range(env.num_envs):
            ambient_light = ambient_min + np.random.rand(3)*(ambient_max - ambient_min)
            ambient_light = np.append(ambient_light, i)
            env.scene.set_ambient_light(ambient_light)
        print("Ambient light randomized.")

    # Randomized directional lights per sub-scene
    if "directional" in options["lighting"]:
        for i in range(env.num_envs):
            # Add config-specified directional lights per sub-scene
            for d in options["lighting"]["directional"]:
                direction_min = np.array(d["direction"][0])
                direction_max = np.array(d["direction"][1])
                direction = direction_min + np.random.rand(3)*(direction_max - direction_min)
                color_min = np.array(d["color"][0])
                color_max = np.array(d["color"][1])
                color = color_min + np.random.rand(3)*(color_max - color_min)
                env.scene.add_directional_light(
                    direction, color, shadow=shadow, shadow_scale=5, shadow_map_size=2048,
                    scene_idxs=[i],
                )
        print("Directional lights randomized.")
            # TODO: Add specified spot and point lights


# Convenience method for randomizing tabletop tasks by building individual subscenes and merging.
def load_table_scenes(env):
    env.tables = []
    env.grounds = []
    env.table_scenes = []

    for i in range(env.num_envs):
        table_scene = TableSceneBuilder(
            env=env, robot_init_qpos_noise=env.robot_init_qpos_noise
        )
        table_scene.build(scene_idxs=[i])
        env.tables.append(table_scene.table)
        env.grounds.append(table_scene.ground)
        env.table_scenes.append(table_scene) # Not for merging, but for initialization

    # Merge table-related actors
    env.table = Actor.merge(env.tables, name="table")
    env.ground = Actor.merge(env.grounds, name="ground")

