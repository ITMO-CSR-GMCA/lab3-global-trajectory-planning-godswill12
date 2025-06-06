# Technical Report: 2D Path Planning Using the Potential Field Method

## 1. Objective
This project implements a 2D path planning algorithm using the Potential Field Method (PFM). The method guides a robot from a fixed start position to a randomly generated goal while avoiding obstacles through the interaction of attractive and repulsive potentials. The resulting path is visualized using both 2D contour and 3D surface plots.

## 2. Problem Statement
Design a Python-based planner that:

Uses a fixed start position and a randomly generated goal.

Defines several static obstacles within a 2D workspace.

Constructs a total potential field across the workspace.

Calculates the gradient of the potential to determine the movement direction.

Plans a collision-free path using gradient descent.

Visualizes the entire planning process.

## 3. Methodology

### 3.1 Workspace Setup

Size: 10 × 10 units (2D grid).

Start Point: (1.0, 1.0)

Goal Point: Randomly generated but constrained away from obstacles.

Obstacles: Seven fixed circular obstacles placed at predetermined positions.

### 3.2 Potential Field Construction
The total potential field at each position is a combination of attractive and repulsive components. The attractive field pulls the robot toward the goal, while the repulsive field pushes it away from obstacles within a specified radius.

### 3.3 Gradient-Based Path Planning
The robot's movement is guided by the negative gradient of the potential field. The planner uses numerical approximation to calculate the gradient and applies a fixed step size to update the robot's position. To overcome local minima, a perturbation mechanism is included to escape stuck positions.

## 4. Implementation Summary

### 4.1 Key Functions

attractive_potential(): Computes pull toward goal

repulsive_potential(): Computes push from obstacles

total_potential(): Combines both fields

plan_path(): Main loop using gradient descent with collision, stuck, and boundary checks

### 4.2 Features

Goal validity check to avoid overlap with obstacles

B-spline interpolation for path smoothing

Diversion detection when nearing obstacles

Visualization includes:

2D contour plots of potential field

3D surface of the potential landscape

Highlighted path, obstacles, influence zones, and diversions

## 5. Results and Visualization

![alt text]({output}.png)


## 6. Observations

Smooth, obstacle-avoiding paths were consistently generated.

Occasional local minima handled by perturbation strategy.

B-spline significantly reduced jaggedness.

Combined attractive and repulsive potentials guided the robot effectively.

## 7. Conclusion
The project demonstrates the effectiveness of the Potential Field Method in 2D path planning. Despite challenges with local minima, the perturbation strategy increases robustness. The planner achieves smooth, valid paths in cluttered environments.

## 8. Recommendations for Improvement

Integrate global planners (e.g., RRT*) for hybrid solutions

Add dynamic obstacle support

Improve local minima escape techniques (e.g., simulated annealing)

Incorporate robot kinematics to reflect realistic motion constraints