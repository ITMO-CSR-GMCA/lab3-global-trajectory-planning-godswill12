import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.colors as colors
from scipy.interpolate import make_interp_spline

def main():
    # Workspace dimensions
    workspace_size = (10, 10)

    # Start point
    start = (1.0, 1.0)

    # Obstacles
    obstacles = [
        (2.5, 3.0),
        (3.0, 7.5),
        (5.0, 5.0),
        (6.5, 2.0),
        (7.0, 6.0),
        (8.0, 4.0),
        (4.0, 8.0)
    ]

    # Potential field parameters
    k_att = 1.0
    k_rep = 15.0
    rho_0 = 1.5

    # Path planning parameters
    step_size = 0.1
    max_steps = 1000
    goal_tolerance = 0.3

    # Generate random goal point not too close to obstacles
    def is_valid_goal(goal, obstacles, min_dist=1.5):
        return all(np.linalg.norm(np.array(goal) - np.array(obs)) >= min_dist for obs in obstacles)

    rng = np.random.default_rng()
    while True:
        goal = tuple(rng.uniform(1, workspace_size[0]-1, size=2))
        if is_valid_goal(goal, obstacles):
            break

    print(f"Random goal point: {goal}")

    def attractive_potential(pos, goal_point, k_attractive_gain):
        return 0.5 * k_attractive_gain * np.linalg.norm(np.array(pos) - np.array(goal_point))**2

    def repulsive_potential(pos, obstacle_list, k_repulsive_gain, influence_radius):
        u_rep = 0
        for obs in obstacle_list:
            dist = np.linalg.norm(np.array(pos) - np.array(obs))
            if dist <= influence_radius:
                if dist < 0.01:
                    dist = 0.01
                u_rep += 0.5 * k_repulsive_gain * (1/dist - 1/influence_radius)**2
        return u_rep

    def total_potential(pos, goal_point, obstacle_list, k_att, k_rep, rho_0):
        return (attractive_potential(pos, goal_point, k_att) +
                repulsive_potential(pos, obstacle_list, k_rep, rho_0))

    # Create grid for potential field visualization
    x_vals = np.linspace(0, workspace_size[0], 100)
    y_vals = np.linspace(0, workspace_size[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = total_potential((X[i, j], Y[i, j]), goal, obstacles, k_att, k_rep, rho_0)

    # Path planning with gradient descent on potential field
    def plan_path(start_point, goal_point, obstacle_list, workspace_dimensions,
                  k_att, k_rep, rho_0, step_size, max_steps, goal_tol):

        path = [np.array(start_point)]
        diversion_points = []
        diversion_segments = []
        current_segment = [np.array(start_point)]
        stuck_counter = 0
        stuck_threshold = 50
        perturbation_strength = 0.5

        for i in range(max_steps):
            current = path[-1]
            if np.linalg.norm(current - goal_point) < goal_tol:
                print(f"Goal reached in {i} steps!")
                break

            eps = max(0.01, min(0.1, np.linalg.norm(current - goal_point)/10))

            dx = (total_potential(current + [eps, 0], goal_point, obstacle_list, k_att, k_rep, rho_0) -
                  total_potential(current - [eps, 0], goal_point, obstacle_list, k_att, k_rep, rho_0)) / (2 * eps)
            dy = (total_potential(current + [0, eps], goal_point, obstacle_list, k_att, k_rep, rho_0) -
                  total_potential(current - [0, eps], goal_point, obstacle_list, k_att, k_rep, rho_0)) / (2 * eps)

            grad = np.array([dx, dy])
            grad_norm = np.linalg.norm(grad)

            near_obstacle = any(np.linalg.norm(current - obs) < rho_0 * 1.1 for obs in obstacle_list)

            if near_obstacle:
                current_segment.append(current.copy())
                if not diversion_points or np.linalg.norm(current - diversion_points[-1]) > rho_0 * 0.3:
                    diversion_points.append(current.copy())
            else:
                if len(current_segment) > 1:
                    diversion_segments.append(np.array(current_segment))
                current_segment = [current.copy()]

            if grad_norm > 0.05:
                direction = -grad / grad_norm
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter >= stuck_threshold:
                    print(f"Stuck at {current}. Applying random perturbation.")
                    angle = np.random.uniform(0, 2*np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)]) * perturbation_strength
                    stuck_counter = 0
                else:
                    direction = np.zeros(2)

            next_pos = current + step_size * direction
            if (next_pos[0] < 0 or next_pos[0] > workspace_dimensions[0] or
                next_pos[1] < 0 or next_pos[1] > workspace_dimensions[1]):
                print(f"Hit boundary at {next_pos}.")
                break

            path.append(next_pos)

        else:
            print(f"Max steps reached. Goal not attained.")

        if len(current_segment) > 1:
            diversion_segments.append(np.array(current_segment))

        if len(path) > 3:
            t = np.linspace(0, 1, len(path))
            spl = make_interp_spline(t, np.array(path), k=3)
            smoothed_path = spl(np.linspace(0, 1, len(path) * 2))
            return smoothed_path, diversion_points, diversion_segments

        return np.array(path), diversion_points, diversion_segments

    # Plan the path
    path, diversion_points, diversion_segments = plan_path(
        start, goal, obstacles, workspace_size, k_att, k_rep, rho_0,
        step_size, max_steps, goal_tolerance
    )

    path_x, path_y = path.T

    # Plotting
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(1, 2, 1)
    norm = colors.LogNorm(vmin=Z.min()+0.01, vmax=Z.max())
    contour = ax1.contourf(X, Y, Z, levels=50, cmap='viridis', norm=norm)
    plt.colorbar(contour, ax=ax1, shrink=0.5, aspect=10, label='Potential')

    for i, obs in enumerate(obstacles):
        ax1.add_patch(Circle(obs, rho_0, color='red', alpha=0.15,
                             label='Obstacle Influence' if i == 0 else ""))
        ax1.scatter(obs[0], obs[1], color='red', s=80, marker='X',
                    label='Obstacle Center' if i == 0 else "")

    ax1.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')

    color_list = plt.cm.rainbow(np.linspace(0, 1, len(diversion_segments)))
    for i, seg in enumerate(diversion_segments):
        seg_x, seg_y = seg.T
        ax1.plot(seg_x, seg_y, color=color_list[i], linewidth=2.5,
                 label=f'Diversion {i+1}' if i < 3 else "", zorder=3)

    if diversion_points:
        div_x, div_y = zip(*diversion_points)
        ax1.scatter(div_x, div_y, color='yellow', s=70, edgecolors='black',
                    linewidths=0.7, label='Diversion Points', zorder=4)

    ax1.scatter(start[0], start[1], color='blue', s=150, marker='o',
                edgecolor='black', linewidth=1.5, label='Start Point', zorder=5)
    ax1.scatter(goal[0], goal[1], color='lime', s=180, marker='*',
                edgecolor='black', linewidth=1.5, label='Goal Point', zorder=5)

    ax1.set(xlabel='X', ylabel='Y', title='2D Path Planning with Potential Fields')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.4, linestyle=':')
    ax1.set_xlim(0, workspace_size[0])
    ax1.set_ylim(0, workspace_size[1])
    ax1.set_aspect('equal')

    # 3D surface plot of potential field
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, norm=norm)
    ax2.set_title('3D Potential Field')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Potential')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
