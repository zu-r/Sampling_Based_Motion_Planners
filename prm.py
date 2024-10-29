import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation
from component_1 import scene_from_file
from collision_checking import check_collision, get_rectangle_corners, generate_random_pose
import multiprocessing as mp

def load_map(filename):
    print(f"Loading map from: {filename}")
    return scene_from_file(filename)

def generate_random_obstacles(num_obstacles, bounds=(-10, 10)):
    obstacles = []

    for i in range(num_obstacles):
        width = random.uniform(0.5, 2.0)
        height = random.uniform(0.5, 2.0)
        x = random.uniform(bounds[0] + width / 2, bounds[1] - width / 2)
        y = random.uniform(bounds[0] + height / 2, bounds[1] - height / 2)
        theta = random.uniform(0, 2 * np.pi)
        obstacles.append((width, height, x, y, theta))
    return obstacles

def is_within_torus(x, y):
    outer_radius = 3.5
    inner_radius = 0.5
    distance_squared = x**2 + y**2
    return (distance_squared <= outer_radius**2) and (distance_squared >= inner_radius**2)

def check_arm_collision(theta1, theta2, environment, L1=2, L2=1.5):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)

    segment1 = [(0, 0), (x1, y1)]
    segment2 = [(x1, y1), (x2, y2)]
    
    for obs in environment:
        w, h, x, y, theta = obs
        obstacle_corners = get_rectangle_corners(w, h, x, y, theta)
        if line_segment_intersects_obstacle(segment1, obstacle_corners) or line_segment_intersects_obstacle(segment2, obstacle_corners):
            return True
    return False

def line_segment_intersects_obstacle(segment, obstacle_corners):
    p1, p2 = segment
    
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

    for i in range(len(obstacle_corners)):
        p3 = obstacle_corners[i]
        p4 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        if line_segments_intersect(p1, p2, p3, p4):
            return True
    return False

def prm(start, goal, environment, robot_type, num_samples=1000, radius=2, k=6):
    roadmap = [start, goal]
    
    for i in range(num_samples):
        while True:
            if robot_type == "freeBody":
                x, y, theta = generate_random_pose()
                robot_corners = get_rectangle_corners(0.5, 0.3, x, y, theta)
                if not check_collision(robot_corners, environment,robot_type=robot_type):
                    roadmap.append((x, y, theta))
                    break
            elif robot_type == "arm":
                theta1, theta2 = random.uniform(-np.pi, np.pi), random.uniform(-np.pi, np.pi)
                x, y = calculate_arm_position(theta1, theta2)

                if is_within_torus(x, y):
                    if not check_arm_collision(theta1, theta2, environment):
                        roadmap.append((theta1, theta2))
                        break

    graph, edges = construct_graph(roadmap, radius, k, environment, robot_type)
    path = a_star_search(graph, start, goal, robot_type)

    if path:
        if robot_type == "arm":
            path = [(float(theta1), float(theta2)) for (theta1, theta2) in path]
        else:
            path = [(float(x), float(y), float(theta)) for (x, y, theta) in path]
        print(f"Path found: {path}")
    else:
        print("No path found.")

    return roadmap, edges, path

def calculate_arm_position(theta1, theta2, L1=2, L2=1.5):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

def angular_distance(node1, node2):
    delta_theta1 = np.abs(node1[0] - node2[0])
    delta_theta2 = np.abs(node1[1] - node2[1])
    return np.sqrt(delta_theta1**2 + delta_theta2**2)

def construct_graph(roadmap, radius, k, environment, robot_type):
    edges = []
    graph = {node: [] for node in roadmap}
    
    for i, node1 in enumerate(roadmap):
        neighbors = []
        for j, node2 in enumerate(roadmap):
            if i != j and angular_distance(node1, node2) <= radius:
                if not edge_collides(node1, node2, environment, robot_type):
                    neighbors.append(node2)
                    edges.append((node1, node2))
                    
        graph[node1] = sorted(neighbors, key=lambda n: angular_distance(node1, n))[:k]
    
    return graph, edges


def interpolate_configuration(node1, node2, num_points, robot_type):
    interpolated_points = []
    for i in range(num_points + 1):
        t = i / num_points
        if robot_type == 'freeBody':
            x = node1[0] + t * (node2[0] - node1[0])
            y = node1[1] + t * (node2[1] - node1[1])
            theta = node1[2] + t * (node2[2] - node1[2])
            interpolated_points.append((x, y, theta))
        elif robot_type == 'arm':
            # Interpolate joint angles directly
            joint1 = node1[0] + t * (node2[0] - node1[0])
            joint2 = node1[1] + t * (node2[1] - node1[1])
            interpolated_points.append((joint1, joint2))
    return interpolated_points

def interpolate_path(path, steps_per_segment,robot_type):
    interpolated_path = []
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        
        if len(p1) == 2:
            p1 = np.append(p1, 0)
        if len(p2) == 2:
            p2 = np.append(p2, 0)
        
        interpolated_segment = interpolate_configuration(p1, p2, steps_per_segment,robot_type)
        interpolated_path.extend(interpolated_segment)
    

    final_point = np.array(path[-1])
    if len(final_point) == 2:
        final_point = np.append(final_point, 0)
    
    interpolated_path.append(final_point)  
    return interpolated_path



def edge_collides(node1, node2, environment, robot_type, num_points=10):
    interpolated_points = interpolate_configuration(node1, node2, num_points, robot_type)
    
    for point in interpolated_points:
        if robot_type == "freeBody":
            x, y, theta = point
            robot_corners = get_rectangle_corners(0.5, 0.3, x, y, theta)
            if check_collision(robot_corners, environment, robot_type):
                return True  
        elif robot_type == "arm":
            theta1, theta2 = point
            if check_arm_collision(theta1, theta2, environment):
                return True  
    return False 


def a_star_search(graph, start, goal, robot_type):
    open_list = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal, robot_type)}

    while open_list:
        current = min(open_list, key=lambda x: x[0])[1]
        open_list = [x for x in open_list if x[1] != current]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph[current]:
            tentative_g_score = g_score[current] + angular_distance(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, robot_type)
                if (f_score[neighbor], neighbor) not in open_list:
                    open_list.append((f_score[neighbor], neighbor))

    return []

def heuristic(a, b, robot_type):
    if robot_type == 'arm':
        return angular_distance(a, b)
    else:
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def visualize_prm(roadmap, edges, path, environment, robot_type):
    fig, ax = plt.subplots()
    
    if robot_type == "arm":
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
    else:
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')

    for obs in environment:
        ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))

    valid_roadmap = []
    
    if robot_type == "arm":
        for point in roadmap:
            if isinstance(point, tuple) and len(point) == 2:
                x, y = calculate_arm_position(point[0], point[1])
                valid_roadmap.append((x, y))
    elif robot_type == "freeBody":
        for point in roadmap:
            if isinstance(point, tuple) and len(point) == 3:
                valid_roadmap.append((point[0], point[1]))

    if valid_roadmap:
        roadmap_x, roadmap_y = zip(*valid_roadmap)
        ax.scatter(roadmap_x, roadmap_y, color='green')

        for node1, node2 in edges:
            if robot_type == "arm" and len(node1) == 2 and len(node2) == 2:
                x1, y1 = calculate_arm_position(node1[0], node1[1])
                x2, y2 = calculate_arm_position(node2[0], node2[1])
                ax.plot([x1, x2], [y1, y2], color='green', alpha=0.5)
            elif robot_type == "freeBody" and len(node1) == 3 and len(node2) == 3:
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='green', alpha=0.5)

    if path:
        if robot_type == "arm":
            path_x, path_y = zip(*[calculate_arm_position(n[0], n[1]) for n in path])
        elif robot_type == "freeBody":
            path_x, path_y = zip(*[(n[0], n[1]) for n in path])

        ax.plot(path_x, path_y, color='red', linewidth=2)


    plt.show()

def visualize_freebody_movement(path, environment):
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    for obs in environment:
        ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))

    robot_width = 0.5
    robot_height = 0.3

    robot_patch = Polygon(np.zeros((4, 2)), color='red', alpha=0.7)
    ax.add_patch(robot_patch)

    def update_freebody(frame):
        x, y, theta = path[frame]
        robot_corners = get_rectangle_corners(robot_width, robot_height, x, y, theta)
        robot_patch.set_xy(robot_corners)
        return robot_patch,

    ani = animation.FuncAnimation(fig, update_freebody, frames=len(path), interval=5, blit=True, repeat=False)
    plt.show()

def visualize_arm_movement(path, environment):
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

    for obs in environment:
        ax.add_patch(Polygon(get_rectangle_corners(*obs), color='blue', alpha=0.5))

    L1, L2 = 2, 1.5

    link1, = ax.plot([], [], color='red', lw=4)
    link2, = ax.plot([], [], color='blue', lw=4)

    def update_arm(frame):
        theta1, theta2 = path[frame][0], path[frame][1]
        print()
        elbow_x = L1 * np.cos(theta1)
        elbow_y = L1 * np.sin(theta1)
        end_effector_x = elbow_x + L2 * np.cos(theta1 + theta2)
        end_effector_y = elbow_y + L2 * np.sin(theta1 + theta2)

        link1.set_data([0, elbow_x], [0, elbow_y])
        link2.set_data([elbow_x, end_effector_x], [elbow_y, end_effector_y])
        return link1, link2

    ani = animation.FuncAnimation(fig, update_arm, frames=len(path), interval=5, blit=True, repeat=False)
    plt.show()

def inverse_kinematics(x, y, L1=2, L2=1.5):
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2

def main():
    parser = argparse.ArgumentParser(description="PRM motion planning for a robot.")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'])
    parser.add_argument('--start', type=float, nargs='+', required=True)
    parser.add_argument('--goal', type=float, nargs='+', required=True)
    parser.add_argument('--map', type=str, required=True)
    
    args = parser.parse_args()

    start, goal = tuple(args.start), tuple(args.goal)

    environment = load_map(args.map)

    roadmap, edges, path = prm(start, goal, environment, args.robot)
    visualize_prm(roadmap, edges, path, environment, args.robot)
    
    if path:
        path = interpolate_path(path,20,args.robot)

        if args.robot == 'arm':
            visualize_arm_movement(path, environment)
        elif args.robot == 'freeBody':
            visualize_freebody_movement(path, environment)
    else:
        print("No valid path to visualize.")

if __name__ == "__main__":
    main()
