import numpy as np
import argparse
import matplotlib.pyplot as plt

def load_configs(filename):
    configs = []
    with open(filename, 'r') as file:
        for line in file:
            configs.append(np.array([float(x) for x in line.split()]))  
    return configs

def angular_distance(theta1, theta2):
    return np.abs((theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi)

def distance_arm(config1, config2):
    return np.sqrt(angular_distance(config1[0], config2[0])**2 + 
                   angular_distance(config1[1], config2[1])**2)

def distance_freebody(config1, config2):
    pos_diff = np.linalg.norm(config1[:2] - config2[:2]) 
    angular_diff = angular_distance(config1[2], config2[2])
    return np.sqrt(pos_diff**2 + angular_diff**2)


def find_nearest_neighbors(target, configs, k, robot_type):
    distances = []
    for config in configs:
        if robot_type == 'arm':
            dist = distance_arm(target, config)
        elif robot_type == 'freeBody':
            dist = distance_freebody(target, config)
        distances.append((dist, config))
    
    distances.sort(key=lambda x: x[0]) 
    return [config for _, config in distances[:k]]

def visualize_arm_neighbors(target, neighbors):
    L1, L2 = 2, 1.5  

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

    def plot_arm(theta1, theta2, color='blue', label=None):
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)

        ax.plot([0, x1], [0, y1], color=color, lw=4, label=label)
        ax.plot([x1, x2], [y1, y2], color=color, lw=4)

    plot_arm(target[0], target[1], color='red', label='Target')

    for i, neighbor in enumerate(neighbors):
        plot_arm(neighbor[0], neighbor[1], color='blue', label=f'Neighbor {i+1}')

    plt.legend()
    plt.show()

def visualize_freebody_neighbors(target, neighbors):
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')

    robot_width = 0.5
    robot_height = 0.3

    def get_rectangle(x, y, theta):
        corners = np.array([[-robot_width / 2, -robot_height / 2],
                            [robot_width / 2, -robot_height / 2],
                            [robot_width / 2, robot_height / 2],
                            [-robot_width / 2, robot_height / 2]])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        rotated_corners = np.dot(corners, rotation_matrix) + np.array([x, y])
        return rotated_corners

    target_rect = get_rectangle(target[0], target[1], target[2])
    ax.add_patch(plt.Polygon(target_rect, color='red', label='Target'))

    for i, neighbor in enumerate(neighbors):
        neighbor_rect = get_rectangle(neighbor[0], neighbor[1], neighbor[2])
        ax.add_patch(plt.Polygon(neighbor_rect, color='blue', label=f'Neighbor {i+1}'))

    plt.legend()
    plt.show()

def visualize_neighbors(target, neighbors, robot_type):
    if robot_type == 'freeBody':
        visualize_freebody_neighbors(target, neighbors)
    elif robot_type == 'arm':
        visualize_arm_neighbors(target, neighbors)

def main():
    parser = argparse.ArgumentParser(description="Find nearest neighbors of robot configurations.")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'], help="Type of robot: 'arm' or 'freeBody'.")
    parser.add_argument('--target', type=float, nargs='+', required=True, help="Target configuration (N values).")
    parser.add_argument('-k', type=int, required=True, help="Number of nearest neighbors to find.")
    parser.add_argument('--configs', type=str, required=True, help="Filename of the configurations file.")
    
    args = parser.parse_args()
    
    configs = load_configs(args.configs)
    target = np.array(args.target)

    neighbors = find_nearest_neighbors(target, configs, args.k, args.robot)

    print("Target Configuration:", target)
    print(f"{args.k} Nearest Neighbors:")
    for neighbor in neighbors:
        print(neighbor)

    visualize_neighbors(target, neighbors, args.robot)

if __name__ == "__main__":
    main()
