import ai2thor.controller
import math
import numpy as np


def rotate_vector_with_degree(v, theta_deg):
    import numpy as np

    # Convert degrees to radians
    theta_rad = np.deg2rad(theta_deg)

    # Define rotation matrix
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]])

    # Rotate vector
    v_rotated = np.dot(R, v)

    return v_rotated


def convert_angle(angle):
    return ((angle + 90) % 180) - 90


def check_orienting(event, object_id):
    robot = event.metadata['agent']

    # Get object position
    object_pos = [obj["position"] for obj in event.metadata["objects"] if obj["objectId"] == object_id][0]

    # Get robot position and forward vector
    robot_pos = robot['position']
    robot_rotation = robot['rotation']
    robot_forward = [0, 1]  # The default forward vector in AI2Thor
    robot_forward = rotate_vector_with_degree(np.asarray(robot_forward), theta_deg=robot_rotation["y"])

    # Calculate vector from robot to object
    object_vec = [object_pos['x'] - robot_pos['x'], object_pos['z'] - robot_pos['z']]

    # Calculate angle between robot forward vector and object vector
    angle = math.acos(np.dot(robot_forward, object_vec) / (np.linalg.norm(robot_forward) * np.linalg.norm(object_vec)))
    angle = abs(convert_angle(angle * (180 / np.pi)) / (180 / np.pi))

    # Check if angle is within threshold
    threshold = math.pi / 4  # 45 degrees
    if angle <= threshold:
        return True
    else:
        return False
