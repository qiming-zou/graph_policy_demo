import os.path

import numpy as np
from zqmtool.other import load_pickle, to_np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
cwd = os.path.dirname(__file__)



def from_onehot(onehot):
    """
    Converts a one-hot embedding to a string.

    Args:
    onehot (list[int]): A one-hot embedding to be converted to a string.

    Returns:
    int: A int corresponding to the one-hot embedding.
    """
    return [i for i, x in enumerate(onehot) if x == 1][0]


def to_onehot(data, num_classes):
    """
    Converts a list of integers to a list of one-hot embeddings.

    Args:
    data (list[int]): A list of integers to be converted to one-hot embeddings.
    num_classes (int): The total number of classes (unique integers) in the data.

    Returns:
    list[list[int]]: A list of one-hot embeddings corresponding to each integer in the data.
    """
    onehot = []
    for d in data:
        onehot.append([int(d == i) for i in range(num_classes)])
    return onehot


def round_pose(x, keep=2):
    x_ = {}
    for k in x.keys():
        x_[k] = round(x[k], keep)

    for k in x.keys():
        if abs(x_[k]) < 0.001:
            x_[k] = 0.0  # to avoid -0.0
    return x_


class ObjectSearchDataset:
    def __init__(self, G_seed_set, target_object, scene_name="FloorPlan1"):
        self.G_seed_set = G_seed_set
        self.scene_name = scene_name
        self.target_object = target_object

        self.node2label = self.collect_node_labels_for_target_object()
        self.all_node_lst = [node for node, labels in self.node2label.items()]
        self.all_node_label_np = to_np([labels for node, labels in self.node2label.items()])
        self.nodeID_cluster_lst, y = self.cluster_node_id_based_on_mutual_information()

        self.out = {}
        self.out["s"] = self.get_s()
        self.out["y"] = y

        """
        out = {
            s: {sid0:[pose4, pose1, ...], sid1: [pose8, ...], ...}
            y:{
                seed0:[0,1,0,0],
                seed1:[0,0,1,0],
                ....
            } # skill_id is the list indice
        }
        """

    def collect_node_labels_for_target_object(self):
        node2label = defaultdict(list)
        for G_seed in self.G_seed_set:
            G = load_pickle(path=os.path.join(cwd, f"../G/{self.scene_name}_{G_seed}.pickle"))
            for node in G.nodes:
                obj_lst = list(G.nodes[node]["object"].values())
                obj_type_lst = [obj["name"] for obj in obj_lst]
                label = int(self.target_object in obj_type_lst)
                node2label[node].append(label)
        return node2label

    def cluster_node_id_based_on_mutual_information(self, ):
        nodeID_cluster_lst = []
        not_clustered_nodeID_set = set()

        y = defaultdict(list)

        for node_id, (node, labels) in enumerate(self.node2label.items()):
            if np.sum(labels) == 0:
                not_clustered_nodeID_set.add(node_id)
                continue
            if np.any([node_id in cluster for cluster in nodeID_cluster_lst]):
                # the node id has been clusterd in one of the cluster
                continue

            labels_np = to_np(labels)
            diff = np.sum(np.abs(self.all_node_label_np - labels_np), axis=1)
            node_ids = list(np.where(diff == 0)[0])  # find the node ids which has the same label in all Graphs

            for enum_id, label_int in enumerate(labels_np):
                y[f"seed{enum_id}"].append(label_int)

            nodeID_cluster_lst.append(node_ids)

        nodeID_cluster_lst.append(
            list(not_clustered_nodeID_set))  # not_clustered_nodeID_set includes the node ids with label 0 in all Graphs

        for seed_str in y.keys():
            y[seed_str].append(0.0)

        return nodeID_cluster_lst, y

    def get_s(self):
        pose_cluster_dict = defaultdict(list)
        for enum_id, nodeID_cluster in enumerate(self.nodeID_cluster_lst):
            for nodeID in nodeID_cluster:
                node = eval(self.all_node_lst[nodeID])
                pos = node["position"]
                rot = node["rotation"]
                pose = [pos["x"], pos["z"]] + [rot["y"]]
                pose_cluster_dict[enum_id].append(pose)
        return pose_cluster_dict

# # checking
# for node_cluster in node_cluster_lst:
#     labels = to_np([node2label[node] for node in node_cluster])
#     other_labels = to_np(
#         [node2label[node] for node_cluster2 in node_cluster_lst for node in node_cluster2 if node not in node_cluster])
#     # numpy checks each row in a cluster are the same
#     assert np.all(labels[0] == labels)
#     # numpy check a row in a cluster is different from each row in another clusters
#     assert labels[0].tolist() not in other_labels.tolist()
#
# # checking via drawing
# state_lst = []
# clusterID_lst = []
# for clusterID, state_cluster in enumerate(state_cluster_lst[:-1]):
#     state_lst += state_cluster
#     clusterID_lst += [clusterID] * len(state_cluster)
#
# state_np = to_np(state_lst)
# state_np -= np.amin(state_np, axis=0)
# state_np /= np.amax(state_np, axis=0)
#
# state_2d_np = PCA(n_components=2).fit_transform(state_np)
# plt.scatter(state_2d_np[:, 0], state_2d_np[:, 1], c=clusterID_lst, alpha=1, cmap=plt.cm.get_cmap("Set1"))
# plt.colorbar()
# plt.show()
