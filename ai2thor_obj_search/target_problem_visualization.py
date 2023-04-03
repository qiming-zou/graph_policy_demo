#!/usr/bin/env python
# coding: utf-8

# In[39]:
from zqmtool.other import load_pickle, to_np
import numpy as np
from utils.controller import CustomController, get_shortest_path_from_pose_to_pose
from utils.visualization_utils import visualize_agent_path, draw_rectangle_on_image, save_img
from utils.bbox import draw_3d_bbox
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from tqdm import trange
from utils.utils import ObjectSearchDataset
from tqdm import tqdm


def set_plt(ax, title=None, times=1,
            xlim=None, ylim=None,
            xlabel=None, ylabel=None,
            hide_tick=True,
            hide_xaxis=True, hide_yaxis=True):
    TIMES = times
    SMALL_SIZE = 6 * TIMES
    MEDIUM_SIZE = 8 * TIMES
    # MEDIUM_SIZE = 12 * TIMES

    ax.spines['bottom'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')

    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', family='serif')
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    if ylim is not None:
        ax.set_ylim([ylim[0] - ylim[1] / 10, ylim[1] + ylim[1] / 10])
        if not hide_tick:
            ax.set_yticks(np.arange(ylim[0], ylim[1] + ylim[2], ylim[2]))
        else:
            ax.set_yticks([])
    if xlim is not None:
        ax.set_xlim([xlim[0] - xlim[1] / 20, xlim[1] + xlim[1] / 20])
        if not hide_tick:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + xlim[2], xlim[2]))
        else:
            ax.set_xticks([])

    if xlabel is not None:
        ax.set_xlabel(xlabel, font='serif', fontsize=MEDIUM_SIZE)

    if ylabel is not None:
        ax.set_ylabel(ylabel, font='serif', fontsize=MEDIUM_SIZE)

    if hide_xaxis:
        ax.get_xaxis().set_visible(False)

    if hide_yaxis:
        ax.get_yaxis().set_visible(False)

    if title is not None:
        ax.set_title(title)


# hyperparameter
scene_name = "FloorPlan1"
tgt_obj = 'Bowl_bb2c17ef'
G_seed_set = list(range(11))

dataset = ObjectSearchDataset(
    target_object=tgt_obj,
    G_seed_set=G_seed_set
).out

# visable_object_type_set = set()
# for G_seed in G_seed_set:
#     path = f"G/{scene_name}_{G_seed}.pickle"
#     G = load_pickle(path=path)
#     for node in G.nodes:
#         for object_i in G.nodes[node]["object"].values():
#             visable_object_type_set.add(object_i["name"])


# ability_attri_dict = {
#     'toggleable': 'isToggled',
#     'breakable': 'isBroken',
#     'canFillWithLiquid': 'isFilledWithLiquid',
#     'dirtyable': 'isDirty',
#     'canBeUsedUp': 'isUsedUp',
#     'cookable': 'isCooked',
#     'sliceable': 'isSliced',
#     'openable': 'isOpen',
# }
# visable_object_type_set = defaultdict(set)
# for seed in trange(2):
#     path = f"G/{scene_name}_{seed}.pickle"
#     G = load_pickle(path=path)
#     for node in G.nodes:
#         for object_i in G.nodes[node]["object"].values():
#             for ability in ability_attri_dict.keys():
#                 if object_i[ability]:
#                     visable_object_type_set[object_i["objectType"]].add(
#                         f'Is {object_i["objectType"]} {ability_attri_dict[ability].replace("is", "")}?'
#                     )
# with open('ObjectType2Question.txt', 'w') as file:
#     for key, value in visable_object_type_set.items():
#         file.write(f'{key}: {", ".join(value)}\n')

for G_seed in tqdm(G_seed_set):
    G_path = f"G/{scene_name}_{G_seed}.pickle"

    # initialize
    G = load_pickle(path=G_path)
    c = CustomController()
    c.reset(seed=G_seed, scene_name=scene_name, top_view=True)

    # get goal nodes and obj bbox
    goal_nodes = []
    obj_pos_lst = []
    for node in G.nodes:
        objs_in_node = G.nodes[node]["object"].values()
        for obj in objs_in_node:
            if obj["name"] == tgt_obj:
                obj_pos_lst.append(obj["position"])
                goal_nodes.append(node)
    for position in obj_pos_lst:
        position_in_top_view = tuple(reversed(c.pos_translator(to_np([position["x"], position["z"]]))))
        frame = draw_rectangle_on_image(img=c.top_view, center=position_in_top_view, width=20)
    if len(obj_pos_lst) == 0:
        c.stop()
        continue

    # start to draw
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))

    ## draw trajectory
    shortest_path = get_shortest_path_from_pose_to_pose(
        G, init_pose=list(G.nodes)[0], tgt_pose=goal_nodes[0]
    )
    traj = visualize_agent_path(shortest_path, c.top_view, pos_translator=c.pos_translator, color_pair_ind=1,
                                show_vis_cone=True)
    axs[0].imshow(traj)
    set_plt(axs[0], title=f"success trajectory")

    ## draw first-person view
    c.reset(seed=G_seed, scene_name=scene_name, top_view=False)
    c.set_agent_to_state(shortest_path[-1])
    event = c.c.last_event
    event = draw_3d_bbox(event=event, constrain=True, target_obj_type=tgt_obj)
    bbox_3d_frame = event.bbox_3d_frame
    axs[1].imshow(bbox_3d_frame)
    set_plt(axs[1], title=f"first-person view")

    ## draw reward distribution
    uv_lst = []
    prob_lst = []
    for node in G.nodes:
        position = eval(node)["position"]
        position_in_top_view = tuple(reversed(c.pos_translator(to_np([position["x"], position["z"]]))))
        uv_lst.append(position_in_top_view)
        prob = float(node in goal_nodes)
        prob_lst.append(prob)

    heatmap = np.ones(c.top_view.shape[:2]) * 0.0
    for i, uv in enumerate(uv_lst):
        heatmap[uv[0], uv[1]] += prob_lst[i]
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    sigma = 8
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    axs[2].imshow(heatmap.T, cmap="Greens")
    axs[2].imshow(c.top_view, alpha=0.2)
    set_plt(axs[2], title=f"reward function")

    ## save image
    plt.subplots_adjust(wspace=0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"examples/{tgt_obj}_{G_seed}.png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()
    c.stop()

    ## draw reward distribution
    fig, ax = plt.subplots(1, 1)
    uv_lst = []
    prob_lst = []
    for skill_id in dataset["s"].keys():
        positions = dataset["s"][skill_id]
        for position in positions:
            position_in_top_view = tuple(reversed(c.pos_translator(to_np(position[:2]))))
            uv_lst.append(position_in_top_view)
            prob_lst.append(dataset["y"][f"seed{G_seed}"][skill_id])

    heatmap = np.ones(c.top_view.shape[:2]) * 0.0
    for i, uv in enumerate(uv_lst):
        heatmap[uv[0], uv[1]] += prob_lst[i]
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
    sigma = 8
    heatmap = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    ax.imshow(heatmap.T, cmap="Greens")
    ax.imshow(c.top_view, alpha=0.2)
    set_plt(ax, title=f"reward function")

    plt.subplots_adjust(wspace=0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f"examples/{tgt_obj}_{G_seed}_dataset_checking.png", bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()
