import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import random
from matplotlib.patches import Polygon
from component_1 import scene_from_file

def load_map(filename):
    return scene_from_file(filename)

def generate_random_pose():
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    theta = random.uniform(0, 2 * np.pi)
    return x, y, theta

def check_collision(robot_corners, obstacles, robot_type, robot_config=None):
    if robot_type == "freeBody":
        return check_collision_freebody(robot_corners, obstacles)
    elif robot_type == "arm":
        return check_collision_arm(robot_config, obstacles)

def check_collision_freebody(robot_corners, obstacles):
    colliding_obstacles = []
    for obs in obstacles:
        w, h, x, y, theta = obs
        obstacle_corners = get_rectangle_corners(w, h, x, y, theta)
        if rectangles_intersect(robot_corners, obstacle_corners):
            colliding_obstacles.append(obs)
    return colliding_obstacles

def check_collision_arm(config, obstacles, L1=2, L2=1.5):
    theta1, theta2 = config
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    
    segment1 = [(0, 0), (x1, y1)]
    segment2 = [(x1, y1), (x2, y2)]

    colliding_obstacles = []
    for obs in obstacles:
        w, h, x, y, theta = obs
        obstacle_corners = get_rectangle_corners(w, h, x, y, theta)
        if (line_segment_intersects_obstacle(segment1, obstacle_corners) or
            line_segment_intersects_obstacle(segment2, obstacle_corners)):
            colliding_obstacles.append(obs)
    return colliding_obstacles

def get_rectangle_corners(w, h, x, y, theta):
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_corners = np.dot(corners, rotation_matrix) + np.array([x, y])
    return rotated_corners

def calculate_arm_position(theta1, theta2, L1=2, L2=1.5):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return [(0, 0), (x1, y1), (x2, y2)]

def rectangles_intersect(rect1, rect2):
    def get_axes(rect):
        edges = rect - np.roll(rect, shift=1, axis=0)
        return edges / np.linalg.norm(edges, axis=1)[:, np.newaxis]

    def project(rect, axis):
        projections = np.dot(rect, axis)
        return np.min(projections), np.max(projections)

    axes = np.concatenate([get_axes(rect1), get_axes(rect2)])
    for axis in axes:
        min1, max1 = project(rect1, axis)
        min2, max2 = project(rect2, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def line_segment_intersects_obstacle(segment, obstacle_corners):
    p1, p2 = segment
    for i in range(len(obstacle_corners)):
        p3 = obstacle_corners[i]
        p4 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        if line_segments_intersect(p1, p2, p3, p4):
            return True
    return False

def line_segments_intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    return (0 <= t <= 1) and (0 <= u <= 1)

def update(frame, env, robot_width, robot_height, robot_type, robot_patches, obstacle_patches):
    colliding_obstacles = []
    if robot_type == 'freeBody':
        x, y, theta = generate_random_pose()
        robot_corners = get_rectangle_corners(robot_width, robot_height, x, y, theta)
        robot_patches['body'].set_xy(robot_corners)
        colliding_obstacles = check_collision(robot_corners, env, robot_type)
    elif robot_type == 'arm':
        theta1, theta2 = random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)
        arm_positions = calculate_arm_position(theta1, theta2)
        robot_patches['link1'].set_data([0, arm_positions[1][0]], [0, arm_positions[1][1]])
        robot_patches['link2'].set_data([arm_positions[1][0], arm_positions[2][0]], [arm_positions[1][1], arm_positions[2][1]])
        colliding_obstacles = check_collision_arm((theta1, theta2), env)

    for obs_patch, obs in zip(obstacle_patches, env):
        obs_patch.set_color('red' if obs in colliding_obstacles else 'green')
    
    return list(robot_patches.values()) + obstacle_patches

def visualize_environment_with_robot(env, robot_type, robot_width=0.5, robot_height=0.3):
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    obstacle_patches = []
    for obs in env:
        w, h, x, y, theta = obs
        obstacle_corners = get_rectangle_corners(w, h, x, y, theta)
        obs_patch = Polygon(obstacle_corners, color='green', alpha=0.5)
        ax.add_patch(obs_patch)
        obstacle_patches.append(obs_patch)

    robot_patches = {}
    if robot_type == 'freeBody':
        robot_patches['body'] = Polygon(np.zeros((4, 2)), color='blue', alpha=0.7)
        ax.add_patch(robot_patches['body'])
    elif robot_type == 'arm':
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        robot_patches['link1'], = ax.plot([], [], color='red', lw=4)
        robot_patches['link2'], = ax.plot([], [], color='blue', lw=4)

    return robot_patches, obstacle_patches, fig

def main():
    parser = argparse.ArgumentParser(description="Collision checking for robot in environment.")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'], help="Type of robot: 'arm' or 'freeBody'.")
    parser.add_argument('--map', type=str, required=True, help="Filename of the map (environment).")

    args = parser.parse_args()
    environment = load_map(args.map)

    robot_patches, obstacle_patches, fig = visualize_environment_with_robot(environment, args.robot)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=10,
        fargs=(environment, 0.5, 0.3, args.robot, robot_patches, obstacle_patches),
        interval=1000,
        repeat=False
    )

    # Save as GIF
    gif_filename = f"component_3_{args.robot}.gif"
    ani.save(gif_filename, writer="pillow")

    print(f"Animation saved as {gif_filename}")
    plt.show()

if __name__ == "__main__":
    main()
