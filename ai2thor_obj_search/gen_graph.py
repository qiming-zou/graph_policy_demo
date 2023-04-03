import os.path
import pickle
from tqdm import tqdm
import networkx as nx
from utils import check_orienting
from utils.controller import CustomController


# ai2thor version 3.0.0
# conda install -c conda-forge ffmpeg

class EnvG():
    def __init__(self, seed):
        self.scene_name = "FloorPlan1"
        self.seed = seed
        self.openness = 0.8
        self.visibilityDistance = 1.0
        self.c = CustomController()
        self.c.reset(seed=seed, scene_name=self.scene_name)
        self.build_nav_open_graph()
        self.add_achieved_object_to_graph()
        self.c.stop()

    def build_nav_open_graph(self, ):
        if os.path.exists(f"G/{self.scene_name}.pickle"):
            self.G = pickle.load(open(f"G/{self.scene_name}.pickle", "rb"))
            return

        G = nx.DiGraph()
        for init_pos, init_rot in tqdm(self.c.reachable_pose, desc="running function 'build_nav_open_graph'"):
            for act in self.c.act_space:
                # set agent to initial pose
                success = self.c.set_agent_to_state(pose={"position": init_pos, "rotation": init_rot})
                if not success:
                    continue

                # take an action
                event = self.c.c.step(action=act)
                position = event.metadata["agent"]["position"]
                rotaiton = event.metadata["agent"]["rotation"]
                # add a navigation edge
                G.add_edge(
                    str({"position": round_pose(init_pos), "rotation": round_pose(init_rot)}),
                    str({"position": round_pose(position), "rotation": round_pose(rotaiton)}),
                    act=act,
                )

                # open a receptacle
                for obj in event.metadata["objects"]:
                    if not (obj["objectId"] in event.instance_masks.keys()):
                        continue
                    if not (obj["openable"]):
                        continue
                    if not obj['receptacle']:
                        continue
                    if not (check_orienting(event, object_id=obj["objectId"])):
                        continue

                    event = self.c.c.step(
                        action="OpenObject",
                        objectId=obj["objectId"],
                        openness=self.openness,
                        forceAction=False
                    )
                    success_to_open = event.metadata["lastActionSuccess"]

                    if success_to_open:
                        G.add_edge(
                            str({"position": round_pose(position), "rotation": round_pose(rotaiton)}),
                            str({"position": round_pose(position), "rotation": round_pose(rotaiton),
                                 "opening": f"{obj['objectId']}"}),
                            act=f"Open {obj['objectId']}",
                        )

                        event = self.c.c.step(
                            action="CloseObject",
                            objectId=obj["objectId"],
                            forceAction=False
                        )
                        success_to_close = event.metadata["lastActionSuccess"]
                        assert success_to_close
                        G.add_edge(
                            str({"position": round_pose(position), "rotation": round_pose(rotaiton),
                                 "opening": f"{obj['objectId']}"}),
                            str({"position": round_pose(position), "rotation": round_pose(rotaiton)}),
                            act=f"Close {obj['objectId']}",
                        )
                    else:
                        continue

        pickle.dump(G, open(f"G/{self.scene_name}.pickle", "wb"))
        self.G = G
        return

    def add_achieved_object_to_graph(self):
        for node in tqdm(self.G.nodes, desc="running function: add_achieved_object_and_frame_to_graph()"):
            node = eval(node)
            success = self.c.set_agent_to_state(node)
            if not success:
                nx.set_node_attributes(self.G, values={str(node): {}}, name="object")
                continue

            event = self.c.c.last_event
            achieved_objects = []
            for obj in event.metadata["objects"]:
                if obj["visible"] and \
                        obj["pickupable"] and \
                        check_orienting(event, object_id=obj["objectId"]) and \
                        obj["objectId"] in event.instance_masks.keys():
                    achieved_objects.append(obj)
            achieved_objects = {item["name"]: item for item in achieved_objects}

            nx.set_node_attributes(self.G, values={str(node): achieved_objects}, name="object")

        pickle.dump(self.G, open(f"G/{self.scene_name}_{self.seed}.pickle", "wb"))


def round_pose(x, keep=2):
    x_ = {}
    for k in x.keys():
        x_[k] = round(x[k], keep)

    for k in x.keys():
        if abs(x_[k]) < 0.001:
            x_[k] = 0.0  # to avoid -0.0
    return x_


if __name__ == "__main__":
    for seed in range(51):
        G = EnvG(seed=seed)
