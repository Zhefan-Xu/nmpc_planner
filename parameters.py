num_states = 9 #[x, y, z, vx, vy, vz, roll, pitch, yaw]
num_inputs = 4 #[T, roll, pitch, yaw_dot]
horizon = 20
time = 8
delta = 0.2 # Maximum Collsion Chance


# Control
g = 9.81
tau_roll = 1
k_roll = 1
tau_pitch = 1
k_pitch = 1


# Optimization
wg = 100000
wg_yaw = 100
ws = 100
wu = 100