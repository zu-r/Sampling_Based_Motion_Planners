import numpy as np
import random
import heapq
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collision_checking import check_collision, get_rectangle_corners
from prm import load_map, edge_collides, calculate_arm_position, interpolate_path, visualize_arm_movement, visualize_freebody_movement  


def sample_free(goal, goal_sample_rate, robot_type):
    if random.random() < goal_sample_rate:
        return goal
    if robot_type == 'freeBody':
        return (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(0, 2 * np.pi))
    elif robot_type == 'arm':
        return (random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi))

def nearest(tree, x_rand, robot_type):
    if robot_type == "arm":
        return min(tree, key=lambda node: np.linalg.norm(np.array(node) - np.array(x_rand)))
    else:
        return min(tree, key=lambda node: np.linalg.norm(np.array(node[:3]) - np.array(x_rand[:3])))

def steer(x_nearest, x_rand, step_size, robot_type):
    if robot_type == 'freeBody':
        direction = np.arctan2(x_rand[1] - x_nearest[1], x_rand[0] - x_nearest[0])
        new_x = x_nearest[0] + step_size * np.cos(direction)
        new_y = x_nearest[1] + step_size * np.sin(direction)
        new_theta = x_nearest[2]
        return (new_x, new_y, new_theta)
    elif robot_type == 'arm':
        direction = np.array(x_rand) - np.array(x_nearest)
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size
        return tuple(np.array(x_nearest) + direction)

def collision_free(x_nearest, x_new, environment, robot_type):
    if robot_type == 'freeBody':
        corners_new = get_rectangle_corners(0.5, 0.3, x_new[0], x_new[1], x_new[2])
        return not check_collision(corners_new, environment,robot_type)
    elif robot_type == 'arm':
        return not edge_collides(x_nearest, x_new, environment, robot_type, num_points=10)

def rrt(start, goal, goal_radius, environment, robot_type, ax=None):
    """RRT algorithm to build a tree and search for a path to the goal."""
    tree = [start]
    parent_map = {start: None}
    max_iterations = 1000
    step_size = 0.9 * goal_radius
    goal_sample_rate = 0.05

    if ax:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        for obs in environment:
            ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))

    for i in range(max_iterations):
        x_rand = sample_free(goal, goal_sample_rate, robot_type)
        x_nearest = nearest(tree, x_rand, robot_type)
        x_new = steer(x_nearest, x_rand, step_size, robot_type)

        if collision_free(x_nearest, x_new, environment, robot_type):
            tree.append(x_new)
            parent_map[x_new] = x_nearest

            if ax:
                if robot_type == 'freeBody':
                    ax.plot([x_nearest[0], x_new[0]], [x_nearest[1], x_new[1]], color='green')
                elif robot_type == 'arm':
                    x_nearest_pos = calculate_arm_position(x_nearest[0], x_nearest[1])
                    x_new_pos = calculate_arm_position(x_new[0], x_new[1])
                    ax.plot([x_nearest_pos[0], x_new_pos[0]], [x_nearest_pos[1], x_new_pos[1]], color='green')
                plt.pause(0.01)

            if robot_type == 'freeBody' and np.linalg.norm(np.array(x_new[:2]) - np.array(goal[:2])) <= goal_radius:
                return tree, parent_map, x_new
            elif robot_type == 'arm' and np.linalg.norm(np.array(x_new) - np.array(goal)) <= goal_radius:
                return tree, parent_map, x_new

    return tree, parent_map, None

def heuristic(node, goal, robot_type):
    """Calculate the heuristic estimated cost from node to goal."""
    return np.linalg.norm(np.array(node[:2]) - np.array(goal[:2]))

def angular_distance(node1, node2):
    """Compute the angular distance between two nodes."""
    return np.linalg.norm(np.array(node1[:2]) - np.array(node2[:2]))

def a_star_search(graph, start, goal, robot_type, goal_radius=0.25):
    """A* search for the optimal path from start to goal in the given graph."""
    open_list = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, robot_type)}
    closed_set = set()

    while open_list:
        current = heapq.heappop(open_list)[1]

        if np.linalg.norm(np.array(current[:2]) - np.array(goal[:2])) <= goal_radius:
            return reconstruct_path(came_from, current)

        closed_set.add(current)

        for neighbor in graph.get(current, []):
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + angular_distance(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, robot_type)
                
                if (f_score[neighbor], neighbor) not in open_list:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  

def reconstruct_path(parent_map, goal_node):
    """Reconstruct the path from start to goal using the parent map."""
    path = [goal_node]
    current = goal_node
    while current in parent_map and parent_map[current] is not None:
        current = parent_map[current]
        path.append(current)
    return path[::-1]

def build_graph(tree, parent_map):
    """Build a graph from the RRT tree using parent-child relationships only."""
    graph = {node: [] for node in tree}
    for child, parent in parent_map.items():
        if parent is not None:
            graph[parent].append(child)
            graph[child].append(parent)
    return graph

def visualize_rrt(tree, parent_map, path, environment, robot_type, ax, show_final_path=False):
    """Visualize the RRT tree with an optional overlay of the final path."""
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')


    for obs in environment:
        ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))


    for node, parent in parent_map.items():
        if parent is not None:
            if robot_type == 'arm':
                start_pos = calculate_arm_position(parent[0], parent[1])
                end_pos = calculate_arm_position(node[0], node[1])
            else:
                start_pos = (parent[0], parent[1])
                end_pos = (node[0], node[1])
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color='green', alpha=0.5)


    if show_final_path and path:
        if robot_type == 'arm':
            path_xy = [calculate_arm_position(node[0], node[1]) for node in path]
        else:
            path_xy = [(node[0], node[1]) for node in path]

        path_x, path_y = zip(*path_xy)
        ax.plot(path_x, path_y, color='red', linewidth=2)

def main():
    parser = argparse.ArgumentParser(description="RRT motion planning for a robot.")
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
    tree, parent_map, final_node = rrt(start, goal, goal_radius, environment, args.robot, ax=ax)

    if final_node:
        rrt_path = reconstruct_path(parent_map, final_node)

        graph = build_graph(tree, parent_map)
        a_star_path = a_star_search(graph, start, goal, args.robot)

        if a_star_path:
            visualize_rrt(tree=tree, parent_map=parent_map, path=a_star_path, environment=environment, robot_type=args.robot, ax=ax, show_final_path=True)
            plt.show()

            interpolated_a_star_path = interpolate_path(a_star_path, 20, args.robot)

            if args.robot == 'arm':
                visualize_arm_movement(interpolated_a_star_path, environment)
            elif args.robot == 'freeBody':
                final_pose = (goal[0], goal[1], goal[2])
                

                last_pose = interpolated_a_star_path[-1]
                
                direction = np.arctan2(final_pose[1] - last_pose[1], final_pose[0] - last_pose[0])
                
                steps = 10
                rpath = [last_pose] 

                for i in range(1, steps + 1):
                    x_interp = last_pose[0] + (final_pose[0] - last_pose[0]) * (i / steps)
                    y_interp = last_pose[1] + (final_pose[1] - last_pose[1]) * (i / steps)
                    theta_interp = last_pose[2] + np.arctan2(np.sin(final_pose[2] - last_pose[2]), np.cos(final_pose[2] - last_pose[2])) * (i / steps)
                    rpath.append((x_interp, y_interp, theta_interp))

                interpolated_a_star_path.extend(rpath)

                visualize_freebody_movement(interpolated_a_star_path, environment)

            plt.show()

        else:
            print("No valid path found using A*.")
    else:
        print("No valid path found using RRT.")

if __name__ == "__main__":
    main()
