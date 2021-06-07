import numpy as np

num_states = 9 #[x, y, z, vx, vy, vz, roll, pitch, yaw]
num_inputs = 4 #[T, roll, pitch, yaw_dot]
horizon = 80
time = 8
delta = 0.3 # Maximum Collsion Chance



# Control
g = 9.81
tau_roll = 1
k_roll = 1
tau_pitch = 1
k_pitch = 1

m = 1.2 #kgg
Thrust_max = 2*m*g
T_max = Thrust_max/m
angle_max = np.pi/6
yaw_rate_max = 0.75

# Optimization
wg = 100000
wg_yaw = 10000
ws = 1000
wu = 100


# T = Trust/m ~= 20, angle < pi/6
Q_u = np.eye(4)
Q_u[0, 0] = 1000
Q_u[1, 1] = 100
Q_u[2, 2] = 100
Q_u[3, 3] = 100

Q_ud = np.eye(4)
Q_ud[0, 0] = 100
Q_ud[1, 1] = 1
Q_ud[2, 2] = 1
Q_ud[3, 3] = 1
