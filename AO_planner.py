import numpy as np
import heapq
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collision_checking import get_rectangle_corners
from prm import load_map, interpolate_path, visualize_arm_movement, visualize_freebody_movement, calculate_arm_position
from rrt import sample_free, nearest, steer, collision_free, visualize_rrt

def calculate_cost(parent_map, start_node, new_node):
    cost = 0
    current = new_node
    while current != start_node:
        parent = parent_map[current]
        cost += np.linalg.norm(np.array(current[:2]) - np.array(parent[:2]))
        current = parent
    return cost

def find_near_nodes(tree, new_node, radius):
    return [node for node in tree if np.linalg.norm(np.array(node[:2]) - np.array(new_node[:2])) <= radius]

def estimate_heuristic(node, goal, robot_type):
    return np.linalg.norm(np.array(node[:2]) - np.array(goal[:2]))

def calculate_angular_distance(node1, node2):
    return np.linalg.norm(np.array(node1[:2]) - np.array(node2[:2]))

def a_star_search(graph, start, goal, robot_type, goal_radius):
    open_list = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: estimate_heuristic(start, goal, robot_type)}
    closed_set = set()

    while open_list:
        current = heapq.heappop(open_list)[1]

        if np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) <= goal_radius:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in graph.get(current, []):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + calculate_angular_distance(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + estimate_heuristic(neighbor, goal, robot_type)

                if (f_score[neighbor], neighbor) not in open_list:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  

def rrt_star(start, goal, goal_radius, environment, robot_type, ax=None):
    tree = [start]
    parent_map = {start: None}
    cost_map = {start: 0}
    max_iterations = 1000
    step_size = 0.95 * goal_radius
    goal_sample_rate = 0.05
    search_radius = 2.5 * step_size

    if ax:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        for obs in environment:
            ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))

    for i in range(max_iterations):
        random_sample = sample_free(goal, goal_sample_rate, robot_type)
        nearest_node = nearest(tree, random_sample, robot_type)
        new_node = steer(nearest_node, random_sample, step_size, robot_type)

        if collision_free(nearest_node, new_node, environment, robot_type):
            near_nodes = find_near_nodes(tree, new_node, search_radius)
            min_node = nearest_node
            min_cost = cost_map[nearest_node] + np.linalg.norm(np.array(new_node[:2]) - np.array(nearest_node[:2]))

            for near_node in near_nodes:
                near_cost = cost_map[near_node] + np.linalg.norm(np.array(new_node[:2]) - np.array(near_node[:2]))
                if near_cost < min_cost and collision_free(near_node, new_node, environment, robot_type):
                    min_node = near_node
                    min_cost = near_cost

            tree.append(new_node)
            parent_map[new_node] = min_node
            cost_map[new_node] = min_cost

            for near_node in near_nodes:
                if near_node != new_node:
                    rewire_cost = cost_map[new_node] + np.linalg.norm(np.array(near_node[:2]) - np.array(new_node[:2]))
                    if rewire_cost < cost_map[near_node] and collision_free(new_node, near_node, environment, robot_type):
                        parent_map[near_node] = new_node
                        cost_map[near_node] = rewire_cost

            if ax:
                if robot_type == 'arm':
                    start_pos = calculate_arm_position(*min_node)
                    end_pos = calculate_arm_position(*new_node)
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='green')
                else:
                    ax.plot([min_node[0], new_node[0]], [min_node[1], new_node[1]], color='green')
                plt.pause(0.01)

            if np.linalg.norm(np.array(new_node[:2]) - np.array(goal[:2])) <= goal_radius:
                return tree, parent_map, new_node

    return tree, parent_map, None

def reconstruct_path(parent_map, goal_node):
    path = [goal_node]
    current = goal_node 
    visited = set()

    while current in parent_map and parent_map[current] is not None:
        if current in visited:
            break
        visited.add(current)
        current = parent_map[current]
        path.append(current)

    return path[::-1] if current in parent_map else []

def build_graph(tree, parent_map):
    graph = {node: [] for node in tree}

    for child, parent in parent_map.items():
        if parent is not None:
            graph[parent].append(child)
            graph[child].append(parent)

    return graph

def main():
    parser = argparse.ArgumentParser(description="RRT* motion planning for a robot.")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--goal_rad', type=float, required=True)
    parser.add_argument('--map', type=str, required=True)

    args = parser.parse_args()

    start, goal = tuple(args.start), tuple(args.goal)
    goal_radius = args.goal_rad
    environment = load_map(args.map)

    fig, ax = plt.subplots()
    tree, parent_map, final_node = rrt_star(start, goal, goal_radius, environment, args.robot, ax=ax)

    if final_node:
        rrt_star_path = reconstruct_path(parent_map, final_node)
        graph = build_graph(tree, parent_map)
        a_star_path = a_star_search(graph, start, goal, args.robot,goal_radius=goal_radius)

        if a_star_path:
            visualize_rrt(tree=tree, parent_map=parent_map, path=a_star_path, environment=environment, robot_type=args.robot, ax=ax, show_final_path=True)
            plt.show()

            interpolated_path = interpolate_path(a_star_path, 20, args.robot)

            if args.robot == 'arm':
                visualize_arm_movement(interpolated_path, environment)
            elif args.robot == 'freeBody':
                visualize_freebody_movement(interpolated_path, environment)
            plt.show()
        else:
            print("No valid path found using A*.")
    else:
        print("No valid path found using RRT*.")

if __name__ == "__main__":
    main()
