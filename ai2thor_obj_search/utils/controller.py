import ai2thor
import cv2
from ai2thor.controller import Controller
import pickle
from zqmtool.other import load_pickle, to_np
import matplotlib.pyplot as plt
from tqdm import trange
import random
import ai2thor
import cv2
import numpy as np
import time
from tqdm import trange, tqdm
import networkx as nx
from .utils import round_pose
from .visualization_utils import position_to_tuple, add_agent_view_triangle, ThorPositionTo2DFrameTranslator

assert ai2thor.__version__ == "3.0.0"


class CustomController():
    def __init__(self):

        self.c = Controller()
        self.set_reachable_pose()
        self.act_space = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight"]

    def reset(self, seed, scene_name, top_view=False):
        self.c = reset_controller(
            controller=self.c, seed=seed, scene_name=scene_name
        )
        self.seed = seed
        self.scene_name = scene_name

        if top_view:
            self.c.step({"action": "ToggleMapView"})
            self.top_view = np.array(self.c.last_event.frame, dtype=np.uint8)

            cam_position = self.c.last_event.metadata["cameraPosition"]
            cam_orth_size = self.c.last_event.metadata["cameraOrthSize"]
            self.pos_translator = ThorPositionTo2DFrameTranslator(
                self.c.last_event.frame.shape, position_to_tuple(cam_position), cam_orth_size
            )

    def draw_obj_rect(self, position):
        position_in_top_view = tuple(reversed(self.pos_translator(to_np([position["x"], position["z"]]))))
        draw_rectangle_on_image(img=self.top_view, center=position_in_top_view, width=20, color=(0, 255, 0))

    def draw_agent_view(self, agent_position):
        self.top_view = add_agent_view_triangle(
            position_to_tuple(agent_position),
            rotation=agent_position["rotation"]['y'],
            frame=self.top_view,
            pos_translator=self.pos_translator,
            scale=0.8,
            opacity=0.8,
        )

    def set_agent_to_state(self, pose):
        inp_position = round_pose(pose["position"])
        inp_rotation = round_pose(pose["rotation"])

        max_try = 3
        for try_count in range(max_try):
            event = self.c.step(
                action="Teleport",
                position=inp_position,
                rotation=inp_rotation,
                horizon=30,
                standing=True
            )
            rotation = round_pose(event.metadata["agent"]["rotation"])
            position = round_pose(event.metadata["agent"]["position"])
            success_teleport = (inp_position == position) and (inp_rotation == rotation)

            for obj in self.c.last_event.metadata["objects"]:
                if not obj["openable"]: continue
                if not obj["isOpen"]: continue
                event = self.c.step(
                    action="CloseObject",
                    objectId=obj["objectId"],
                    forceAction=False
                )

            success_open = True
            if "opening" in pose.keys():
                event = self.c.step(
                    action="OpenObject",
                    objectId=pose["opening"],
                    forceAction=False
                )
                success_open = event.metadata["lastActionSuccess"]

            if success_open and success_teleport:
                break
            else:
                self.reset(seed=self.seed, scene_name=self.scene_name)

        if try_count >= (max_try - 1):
            print("failed to reset agent state")
            return False

        return True

    def set_reachable_pose(self):
        event = self.c.step(action="GetReachablePositions")
        self.reachable_positions = event.metadata["actionReturn"]
        self.reachable_rotations = [{"x": 0.0, "y": y, "z": 0.0} for y in [0.0, 90.0, 180.0, 270.0]]
        self.reachable_pose = [[p, r] for p in self.reachable_positions for r in self.reachable_rotations]

    def stop(self):
        self.c.stop()


def get_shortest_path_from_pose_to_pose(G, init_pose, tgt_pose):
    import networkx as nx
    shortest_path = nx.shortest_path(G, init_pose, tgt_pose)
    shortest_path = [eval(node) for node in shortest_path]
    return shortest_path


def reset_controller(controller, scene_name, seed):
    event = controller.reset(scene_name=scene_name, renderInstanceSegmentation=True)
    assert event.metadata["lastActionSuccess"]
    event = controller.step(
        action="InitialRandomSpawn",
        randomSeed=seed,
        forceVisible=False,
        numPlacementAttempts=5,
        placeStationary=True,
        excludedReceptacles=["Cabinet", "Microwave"],
    )
    assert event.metadata["lastActionSuccess"]
    return controller
