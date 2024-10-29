import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collision_checking import check_collision, get_rectangle_corners
from prm import load_map, interpolate_path, visualize_freebody_movement

def sample_free_space(goal, goal_sample_rate):
    if random.random() < goal_sample_rate:
        return goal
    return (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-np.pi, np.pi))

def find_nearest_node(tree, random_sample):
    return min(tree, key=lambda node: np.linalg.norm(np.array(node[:2]) - np.array(random_sample[:2])))

def steer_toward_nearest(nearest_node, random_sample, step_size, velocity, length, steering_angle):

    direction = np.arctan2(random_sample[1] - nearest_node[1], random_sample[0] - nearest_node[0])
    
    angle_diff = direction - nearest_node[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  
    
    angle_change = np.clip(angle_diff, -steering_angle, steering_angle)

    beta = np.arctan(0.5 * np.tan(angle_change))
    
    new_x = nearest_node[0] + step_size * np.cos(nearest_node[2] + beta)
    new_y = nearest_node[1] + step_size * np.sin(nearest_node[2] + beta)
    
    new_theta = nearest_node[2] + (2 * step_size / length) * np.sin(beta)
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  

    return (new_x, new_y, new_theta)



def is_collision_free(nearest_node, new_node, environment,robot_type):
    new_corners = get_rectangle_corners(0.5, 0.3, new_node[0], new_node[1], new_node[2])
    return not check_collision(new_corners, environment,robot_type)

def rrt_algorithm(start_node, goal_node, goal_radius, environment, velocity, steering_angle, ax, robot_type):
    tree = [start_node]
    parent_map = {start_node: None}
    max_iterations = 1000
    step_size = 0.90 * goal_radius
    goal_sample_rate = 0.05
    length = 0.5

    if ax:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        for obstacle in environment:
            ax.add_patch(Polygon(get_rectangle_corners(*obstacle), color='blue', alpha=0.5))

    for i in range(max_iterations):
        random_sample = sample_free_space(goal_node, goal_sample_rate)
        nearest_node = find_nearest_node(tree, random_sample)
        new_node = steer_toward_nearest(nearest_node, random_sample, step_size, velocity, length, steering_angle)

        if new_node and is_collision_free(nearest_node, new_node, environment,robot_type):
            tree.append(new_node)
            parent_map[new_node] = nearest_node

            if ax:
                ax.plot([nearest_node[0], new_node[0]], [nearest_node[1], new_node[1]], color='green')
                plt.pause(0.01)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(goal_node[:2])) <= goal_radius:
                return tree, parent_map, new_node

    return tree, parent_map, None

def reconstruct_path_from_parent_map(parent_map, goal_node):
    path = [goal_node]
    current = goal_node
    while current in parent_map and parent_map[current] is not None:
        current = parent_map[current]
        path.append(current)
    return path[::-1]

def build_graph_from_tree(tree, parent_map):
    graph = {node: [] for node in tree}
    for child, parent in parent_map.items():
        if parent is not None:
            graph[parent].append(child)
            graph[child].append(parent)
    return graph

def visualize_rrt_tree(tree, parent_map, path, environment, ax, show_final_path=False):
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    for obstacle in environment:
        ax.add_patch(Polygon(get_rectangle_corners(*obstacle), color='blue', alpha=0.5))

    for node, parent in parent_map.items():
        if parent is not None:
            ax.plot([parent[0], node[0]], [parent[1], node[1]], color='green', alpha=0.5)

    if show_final_path and path:
        path_x, path_y = zip(*[(node[0], node[1]) for node in path])
        ax.plot(path_x, path_y, color='red', linewidth=2)

def main():
    parser = argparse.ArgumentParser(description="RRT motion planning for a car-like robot.")
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--velocity', type=float, required=True, help="Constant forward velocity of the car.")
    parser.add_argument('--steering_angle', type=float, required=True, help="Maximum steering angle of the car in radians.")

    args = parser.parse_args()
    start_node, goal_node = tuple(args.start), tuple(args.goal)
    goal_radius = args.goal_rad
    velocity = args.velocity
    steering_angle = args.steering_angle

    environment = load_map(args.map)
    fig, ax = plt.subplots()
    tree, parent_map, final_node = rrt_algorithm(start_node, goal_node, goal_radius, environment, velocity, steering_angle, ax=ax,robot_type="freeBody")

    if final_node:
        rrt_path = reconstruct_path_from_parent_map(parent_map, final_node)
        graph = build_graph_from_tree(tree, parent_map)
        visualize_rrt_tree(tree=tree, parent_map=parent_map, path=rrt_path, environment=environment, ax=ax, show_final_path=True)
        plt.show()

        interpolated_rrt_path = interpolate_path(rrt_path, 20, 'freeBody')
        visualize_freebody_movement(interpolated_rrt_path, environment)
        plt.show()
    else:
        print("No valid path found using RRT.")

if __name__ == "__main__":
    main()
