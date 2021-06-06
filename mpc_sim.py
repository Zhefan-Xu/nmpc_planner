from casadi import *
import matplotlib.pyplot as plt
import numpy as np
from obstacles import *
import matplotlib.patches as patches

# Robot size and uncertainy is ZERO

num_states = 9 #[x, y, z, vx, vy, vz, roll, pitch, yaw]
num_inputs = 4 #[T, roll, pitch, yaw_dot]
horizon = 40
time = 3
delta = 0.1 # Maximum Collsion Chance
g = 9.81
tau_roll = 1
k_roll = 1
tau_pitch = 1
k_pitch = 1

# we first consider 2D case: A quadcopter moving in 2D plane (z = constant, vz = 0)

# optimizer
opti = casadi.Opti()

X = opti.variable(num_states, horizon+1)
U = opti.variable(num_inputs, horizon)

# Define each variable
x = X[0, :]
y = X[1, :]
z = X[2, :]
vx = X[3, :]
vy = X[4, :]
vz = X[5, :]
roll = X[6, :]
pitch = X[7, :]
yaw = X[8, :]

T = U[0, :]
roll_d = U[1, :]
pitch_d = U[2, :]
yaw_dot_d = U[3, :]

pos = X[0:3, :]

pos_start = np.array([0, 0, 0])
orientation_start = np.array([0, 0, 0])
pos_goal = np.array([7, 9, 0]).reshape(-1, 1)
orientation_goal = np.array([0, 0, np.pi/2])

# print(pos_goal)
trajectory_desired = np.repeat(pos_goal, horizon+1,axis=1)
# print(trajectory_desired[0])

def make_dynamics_function_rhs():
	states = MX.sym("states", num_states, 1)
	inputs = MX.sym("inputs", num_inputs, 1)

	x = states[0]
	y = states[1]
	z = states[2]
	vx = states[3]
	vy = states[4]
	vz = states[5]
	roll = states[6]
	pitch = states[7]
	yaw = states[8]

	T = inputs[0]
	roll_d = inputs[1]
	pitch_d = inputs[2]
	yaw_dot_d = inputs[3]

	# right hand side of dynamics equation
	rhs = vertcat(
			vx,
			vy,
			vz,
			T*cos(roll)*sin(pitch)*cos(yaw) + T*sin(roll)*sin(yaw),
			T*cos(roll)*sin(pitch)*sin(yaw) - T*sin(roll)*cos(yaw),
			T*cos(roll)*cos(pitch) - g,
			(1/tau_roll) * (k_roll*roll_d - roll),
			(1/tau_pitch) * (k_pitch*pitch_d - pitch),
			yaw_dot_d
		   )
	dynamics_func = Function("dynamics", [states, inputs], [rhs])
	return dynamics_func

dynamics_func = make_dynamics_function_rhs()
print(dynamics_func)



def make_dynamics_update_function(dynamics_func):
	states = MX.sym("states", num_states, 1)
	inputs = MX.sym("inputs", num_inputs, 1)
	ode = {'x': states, 'p':inputs, 'ode':dynamics_func(states, inputs)}

	rk_opts = {'t0':0, 'tf': time/horizon, 'number_of_finite_elements': 1}

	dynamics_intg = integrator('dynamics_integrator', 'rk', ode, rk_opts)
	# dynamics_intg = integrator('dynamics_integrator', 'idas', ode)
	result = dynamics_intg(x0=states, p=inputs)
	dynamics_update_function = Function('dynamics_update_function', [states, inputs], [result['xf']])
	return dynamics_update_function

dynamics_update_function = make_dynamics_update_function(dynamics_func)

# Set constrains
for i in range(horizon):
	# dyanmics constain:
	opti.subject_to(X[:, i+1] == dynamics_update_function(X[:, i], U[:, i]))
	opti.subject_to(T[i] == g/(cos(roll[i])*cos(pitch[i]))) # no vertical net force

	# Obstacle constain:
	for obs_pos, size, uncertainty in zip(static_obstacle, static_obstacle_size, static_obstacle_uncertainty):
		# Transform matrix (to ellipse)
		omega_sqrt = np.array([
					      [1/((size[0]/2 * np.sqrt(2))), 0],
						  [0, 1/((size[1]/2 * np.sqrt(2)))]
							])

		print(obs_pos)
		cov_o = np.eye(2)
		cov_o[0, 0] = uncertainty[0]
		cov_o[1, 1] = uncertainty[1]
		# cov_o[2, 2] = uncertainty[2]
		cov_o_hat = omega_sqrt.T @ cov_o @ omega_sqrt

		cov_total_hat = cov_o

		current_pos = pos[:, i]	# current position
		current_pos_hat = omega_sqrt @ current_pos[0:2]

		obs_pos_hat = omega_sqrt @ np.array(obs_pos[0:2]).reshape(-1, 1)
		v_io_hat = current_pos_hat - obs_pos_hat 


		unit_io_hat = v_io_hat/norm_2(v_io_hat) # a_io

		c = erfinv(1-2*delta) * np.sqrt(2*unit_io_hat.T @ cov_total_hat @ unit_io_hat)

		opti.subject_to(unit_io_hat.T @ v_io_hat - 1 >= c)

	
opti.subject_to(X[0:3, 0] == pos_start)
opti.subject_to(X[6:9, 0] == orientation_start[0:3])

# Set the objective equation
opti.minimize(1000000*sumsqr(pos[:, -1] - trajectory_desired[:, -1]) + 1000 * sumsqr(yaw[-1] - orientation_goal[2]) + 100*sumsqr(pos[:, 0:-1] - trajectory_desired[:, 0:-1]) + 100*sumsqr(U**2))

p_opts = {"expand":True}
s_opts = {"max_iter": 100}
opti.solver("ipopt")

time_arr = [i for i in range(horizon+1)]
sol = opti.solve()
sol_x = sol.value(x)
sol_y = sol.value(y)
sol_z = sol.value(z)
sol_yaw = sol.value(yaw)

sol_X = sol.value(X)
# sol_T = sol.value(T)
# sol_roll = sol.value(roll)
sol_yaw_dot_d = sol.value(yaw_dot_d)

fig, ax = plt.subplots()
# plot for obstacles:
for obs_pos, size in zip(static_obstacle, static_obstacle_size):
	xo, yo = obs_pos[0] - size[0]/2, obs_pos[1] - size[1]/2
	rect = patches.Rectangle((xo, yo), size[0], size[1], linewidth=1, edgecolor='r', facecolor='none')
	ellipse = patches.Ellipse((obs_pos[0], obs_pos[1]), 2* size[0]/2 * np.sqrt(2), 2* size[1]/2 * np.sqrt(2), linewidth=1, edgecolor='g', facecolor='none')
	ax.add_patch(rect)
	ax.add_patch(ellipse)

ax.set_xlim([-5, 11])
ax.set_ylim([-5, 11])
# ax.xlim([-1,8])å
# ax.ylim([-1,8])
for i in range(len(sol_x)):
	# ax.plot(sol_x[i], sol_y[i], marker=[3, 0, sol_yaw[i]*180/np.pi])
	if i != len(sol_x) -1:
		ax.plot(sol_x[i:i+2], sol_y[i:i+2], '-')
	ax.arrow(sol_x[i], sol_y[i], 0.3*cos(sol_yaw[i]), 0.3*sin(sol_yaw[i]), width=0.01)

	plt.pause(0.1)

# print(sol_yaw)
# print(sol_yaw_dot_d)
# print(sol_X)
# plt.plot(time_arr, sol_T)
# plt.figure(2)
# plt.plot(time_arr, sol_x)
# plt.figure(3)
# plt.plot(time_arr, sol_y)
plt.show()

