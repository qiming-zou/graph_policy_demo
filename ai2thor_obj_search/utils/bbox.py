import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .orienting import check_orienting
import matplotlib.pyplot as plt
import os

cwd = os.path.dirname(__file__)


def get_rotation_matrix(agent_rot):
    #######
    # Construct the rotation matrix. Ref: https://en.wikipedia.org/wiki/Rotation_matrix
    #######

    r_y = np.array([[np.cos(np.radians(agent_rot["y"])), 0, np.sin(np.radians(agent_rot["y"]))],
                    [0, 1, 0],
                    [-np.sin(np.radians(agent_rot["y"])), 0, np.cos(np.radians(agent_rot["y"]))]])
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(np.radians(agent_rot["x"])), -np.sin(np.radians(agent_rot["x"]))],
                    [0, np.sin(np.radians(agent_rot["x"])), np.cos(np.radians(agent_rot["x"]))]])
    r = r_x @ r_y
    return r


def project_to_agent_coordinate(pos, agent_pos, r):
    #######
    # Project a position from the world coordinate to the agent coordinate.
    #######

    pos_diff = pos - agent_pos
    # since AI2THOR is left-handed coordinate system, we need to turn it to the right-handed to use the rotation matrix
    pos_diff[2] *= -1
    new_pos = r @ pos_diff
    # turn back to the left-handed coordinate system
    new_pos[2] *= -1
    return new_pos


def project_to_2d(pos, half_fov, w, h):
    #######
    # Project a given 3D position to 2D space.
    #######

    pos_2d = [pos[0] / (pos[2] * np.tan(np.radians(half_fov))),
              pos[1] / (pos[2] * np.tan(np.radians(half_fov)))]

    # x-axis
    x = int(w * ((pos_2d[0] + 1.0) / 2.0))
    # y-axis
    y = int(h * (1 - ((pos_2d[1] + 1.0) / 2.0)))
    return [x, y]


def draw_3d_bbox(event, constrain=False, target_obj_type=None):
    #######
    # Draw the 3D bbox in 2D RGB image by first construct the rotation matrix and get agent position by the agent pose,
    # then filter out the objects which are not visible to the agent.
    # Finally, project the 3D bbox to 2D space and draw it on the 2D RGB image and return the event dict with image.
    #######

    # get the 2D image width and height
    w, h = event.metadata["screenWidth"], event.metadata["screenHeight"]

    # get the FOV
    half_fov = event.metadata["fov"] / 2

    # get the camera rotation matrix
    agent_rot = event.metadata["agent"]["rotation"]
    agent_rot["x"] = event.metadata["agent"]["cameraHorizon"]
    rotation_matrix = get_rotation_matrix(agent_rot)

    # get the camera 3D position
    agent_pos = np.array([event.metadata["cameraPosition"]["x"],
                          event.metadata["cameraPosition"]["y"],
                          event.metadata["cameraPosition"]["z"]])

    # get the 2D RGB image and allocate a drawer
    img = Image.fromarray(event.frame, "RGB")
    draw = ImageDraw.Draw(img)

    # iterate over all objects in the scene
    # first classify if the object is in the view by rotated z position and instance segmentation
    # then draw the 3D bbox in the 2D RGB image
    for obj in event.metadata["objects"]:
        if constrain:
            if not (obj["pickupable"] and \
                    check_orienting(event, object_id=obj["objectId"]) and \
                    obj["objectId"] in event.instance_masks.keys() and \
                    obj["visible"]):
                continue
        if target_obj_type is not None:
            if not (obj["name"] == target_obj_type):
                continue

        # get object 3D position and rotate it to the agent coordinate
        pos = np.array([obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]])
        new_pos = project_to_agent_coordinate(pos, agent_pos, rotation_matrix)

        # classify is the object is in front of the agent
        if new_pos[2] > 0:
            # classify if the object is seen by the agent (not occluded by other objects)
            if obj["objectId"] in event.instance_masks.keys():
                # don't draw the floor and ceiling objects
                if "Floor" in obj["objectId"] or "Ceiling" in obj["objectId"]:
                    if "Lamp" not in obj["objectId"]:
                        continue

                # get the object color from the instance segmentation
                color = event.object_id_to_color[obj["objectId"]]

                # get the 3D bbox center and size
                vertices, valid = [], []
                if not isinstance(obj["objectOrientedBoundingBox"], type(None)):
                    # get the 3D bbox 8 vertices
                    corner_points = obj["objectOrientedBoundingBox"]["cornerPoints"]

                    # project vertices to 2D image coordinate
                    for point in corner_points:
                        new_point = project_to_agent_coordinate(point, agent_pos, rotation_matrix)
                        if new_point[2] > 0:
                            valid.append(True)
                        else:
                            valid.append(False)
                        new_point_2d = project_to_2d(new_point, half_fov, w, h)
                        vertices.append(new_point_2d)

                    # get the 3D bbox 12 lines
                    lines = [[vertices[0], vertices[1]],
                             [vertices[2], vertices[3]],
                             [vertices[0], vertices[3]],
                             [vertices[1], vertices[2]],
                             [vertices[4], vertices[5]],
                             [vertices[6], vertices[7]],
                             [vertices[4], vertices[7]],
                             [vertices[5], vertices[6]],
                             [vertices[2], vertices[6]],
                             [vertices[3], vertices[7]],
                             [vertices[1], vertices[5]],
                             [vertices[0], vertices[4]]]
                    valid_lines = [valid[0] * valid[1],
                                   valid[2] * valid[3],
                                   valid[0] * valid[3],
                                   valid[1] * valid[2],
                                   valid[4] * valid[5],
                                   valid[6] * valid[7],
                                   valid[4] * valid[7],
                                   valid[5] * valid[6],
                                   valid[2] * valid[6],
                                   valid[3] * valid[7],
                                   valid[1] * valid[5],
                                   valid[0] * valid[4]]
                else:
                    if "cornerPoints" in obj["axisAlignedBoundingBox"].keys():
                        # get the 3D bbox 8 vertices
                        corner_points = obj["axisAlignedBoundingBox"]["cornerPoints"]
                    else:
                        # get the 3D bbox 8 vertices from bbox center and size
                        center = np.array([obj["axisAlignedBoundingBox"]["center"]["x"],
                                           obj["axisAlignedBoundingBox"]["center"]["y"],
                                           obj["axisAlignedBoundingBox"]["center"]["z"]])
                        size = np.array([obj["axisAlignedBoundingBox"]["size"]["x"],
                                         obj["axisAlignedBoundingBox"]["size"]["y"],
                                         obj["axisAlignedBoundingBox"]["size"]["z"]])
                        corner_points = []
                        for i in range(2):
                            pos_x = np.array(center)
                            pos_x[0] = pos_x[0] - (size[0] / 2) + (i * size[0])
                            for j in range(2):
                                pos_y = np.array(pos_x)
                                pos_y[1] = pos_y[1] - (size[1] / 2) + (j * size[1])
                                for k in range(2):
                                    pos_z = np.array(pos_y)
                                    pos_z[2] = pos_z[2] - (size[2] / 2) + (k * size[2])
                                    corner_points.append(pos_z)

                    # project vertices to 2D image coordinate
                    for point in corner_points:
                        new_point = project_to_agent_coordinate(point, agent_pos, rotation_matrix)
                        if new_point[2] > 0:
                            valid.append(True)
                        else:
                            valid.append(False)
                        new_point_2d = project_to_2d(new_point, half_fov, w, h)
                        vertices.append(new_point_2d)

                    # get the 3D bbox 12 lines
                    lines = [[vertices[0], vertices[1]],
                             [vertices[2], vertices[3]],
                             [vertices[0], vertices[2]],
                             [vertices[1], vertices[3]],
                             [vertices[4], vertices[5]],
                             [vertices[6], vertices[7]],
                             [vertices[4], vertices[6]],
                             [vertices[5], vertices[7]],
                             [vertices[2], vertices[6]],
                             [vertices[3], vertices[7]],
                             [vertices[1], vertices[5]],
                             [vertices[0], vertices[4]]]
                    valid_lines = [valid[0] * valid[1],
                                   valid[2] * valid[3],
                                   valid[0] * valid[2],
                                   valid[1] * valid[3],
                                   valid[4] * valid[5],
                                   valid[6] * valid[7],
                                   valid[4] * valid[6],
                                   valid[5] * valid[7],
                                   valid[2] * valid[6],
                                   valid[3] * valid[7],
                                   valid[1] * valid[5],
                                   valid[0] * valid[4]]

                lines = np.array(lines)
                lines = np.reshape(lines, (-1, 4))
                valid_lines = np.array(valid_lines)
                valid_lines = np.reshape(valid_lines, (-1, 1))

                # draw the 3D bbox 12 lines in the 2D RGB image
                for iii, line in enumerate(lines):
                    if valid_lines[iii]:
                        color = (0, 255, 0)
                        draw.line((line[0], line[1], line[2], line[3]), fill=color, width=2)

                # Define title text and font
                title_text = obj["objectType"]
                title_font = ImageFont.truetype(os.path.join(cwd, 'dejavu-serif/DejaVuSerif.ttf'), size=32)
                # Calculate position of title text
                title_x = np.min(vertices, axis=0)[0]
                title_y = np.min(vertices, axis=0)[1] - title_font.getsize(title_text)[1]
                # Draw title text on image
                draw.text((title_x, title_y), title_text, font=title_font, fill=(255, 255, 255))

    # store the result back to the event
    bbox_frame = np.array(img)
    event.bbox_3d_frame = bbox_frame
    return event
