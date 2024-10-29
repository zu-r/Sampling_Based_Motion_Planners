import numpy as np
import matplotlib.pyplot as plt
import random

def generate_environment(number_of_obstacles):
    environment = []
    for i in range(number_of_obstacles):
        w, h = random.uniform(0.5, 2), random.uniform(0.5, 2)
        x = random.uniform(-10, 10) 
        y = random.uniform(-10, 10)  
        theta = random.uniform(0, 2 * np.pi)
        obstacle = (w, h, x, y, theta)
        environment.append(obstacle)
    print(environment)
    return environment

def scene_to_file(env, filename):
    with open(filename, 'w') as f:
        for obstacle in env:
            f.write(' '.join(map(str, obstacle)) + '\n')  

def scene_from_file(filename):
    environment = []
    with open(filename, 'r') as f:
        for line in f:
            obstacle = tuple(map(float, line.split())) 
            environment.append(obstacle)
    return environment

def visualize_scene(env):
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10) 
    ax.set_ylim(-10, 10)  
    ax.set_aspect('equal')

    for o in env:
        w, h = o[0], o[1]
        x, y = o[2], o[3]
        theta = o[4]

        corners = np.array([[-w / 2, -h / 2], 
                            [w / 2, -h / 2], 
                            [w / 2, h / 2], 
                            [-w / 2, h / 2]])

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        
        rotated_corners = np.dot(corners, rotation_matrix) + np.array([x, y])

        rect = plt.Polygon(rotated_corners, color='green', alpha=0.5)
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    # temp = 1
    # for i in range(2, 7):
    #     env = generate_environment(i * 2)  
    #     filename = f'environment_{temp}.txt'
    #     scene_to_file(env, filename)
    #     loaded_env = scene_from_file(filename)
    #     visualize_scene(loaded_env)
    #     temp += 1
    for i in range(1,6):
        filename = f'environment_{i}.txt'
        loaded_env = scene_from_file(filename)
        visualize_scene(loaded_env)
