import numpy as np
from casadi import *
from parameters import *
from obstacles import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Base Class
class nmpc_planner:
	opti = casadi.Opti()
	X = opti.variable(num_states, horizon+1)
	U = opti.variable(num_inputs, horizon)

	def __init__(self):
		self.seperate_variable_name()
		self.dynamics_update_function = self.make_dynamics_model()




	def seperate_variable_name(self):
		# Define each variable across time
		self.x = self.X[0, :]
		self.y = self.X[1, :]
		self.z = self.X[2, :]
		self.vx = self.X[3, :]
		self.vy = self.X[4, :]
		self.vz = self.X[5, :]
		self.roll = self.X[6, :]
		self.pitch = self.X[7, :]
		self.yaw = self.X[8, :]

		self.pos = self.X[0:3, :]

		self.T = self.U[0, :]
		self.roll_d = self.U[1, :]
		self.pitch_d = self.U[2, :]
		self.yaw_dot_d = self.U[3, :]


	def make_dynamics_model(self):
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

		ode = {'x': states, 'p':inputs, 'ode':dynamics_func(states, inputs)}

		rk_opts = {'t0':0, 'tf': time/horizon, 'number_of_finite_elements': 1}

		dynamics_intg = integrator('dynamics_integrator', 'rk', ode, rk_opts)
		# dynamics_intg = integrator('dynamics_integrator', 'idas', ode)
		result = dynamics_intg(x0=states, p=inputs)
		dynamics_update_function = Function('dynamics_update_function', [states, inputs], [result['xf']])
		return dynamics_update_function

	def generate_trajectory(self, start, goal, plot=False):
		self.pos_start = np.array(start[0:3]).reshape(-1, 1)
		self.orientation_start = np.array(start[3:6]).reshape(-1, 1)
		self.pos_goal = np.array(goal[0:3]).reshape(-1, 1)
		self.orientation_goal = np.array(goal[3:6]).reshape(-1, 1)
		self.trajectory_desired = np.repeat(self.pos_goal, horizon+1,axis=1)

		self.set_optimize_function()
		self.set_constrains()
		# s_opts = {"max_iter": 300}
		# p_opts = {"expand":False}
		self.opti.solver("ipopt")
		sol = self.opti.solve()
		sol_X = sol.value(self.X)
		self.solution = sol_X
		self.inputs_sol = sol.value(self.U)

		# clear optmizer
		self.opti = casadi.Opti()
		self.X = self.opti.variable(num_states, horizon+1)
		self.U = self.opti.variable(num_inputs, horizon)
		self.seperate_variable_name()

		if not plot:
			return sol_X
		else:
			self.fig, self.ax = plt.subplots()
			self.first_time = True
			self.ax.set_xlim([ENV_X_MIN*1.1, ENV_X_MAX*1.1])
			self.ax.set_ylim([ENV_Y_MIN*1.1, ENV_Y_MAX*1.1])
			
			sol_x = sol_X[0, :]
			sol_y = sol_X[1, :]
			sol_yaw = sol_X[8, :]
			plt.plot(goal[0], goal[1], "*")
			if self.first_time:
				for obs_pos, size in zip(static_obstacle, static_obstacle_size):
					xo, yo = obs_pos[0] - size[0]/2, obs_pos[1] - size[1]/2
					rect = patches.Rectangle((xo, yo), size[0], size[1], linewidth=1, edgecolor='r', facecolor='none')
					ellipse = patches.Ellipse((obs_pos[0], obs_pos[1]), 2* size[0]/2 * np.sqrt(2), 2* size[1]/2 * np.sqrt(2), linewidth=1, edgecolor='g', facecolor='none')
					self.ax.add_patch(rect)
					self.ax.add_patch(ellipse)
				self.first_time = False

			for i in range(len(sol_x)):
				if i != len(sol_x) -1:
					self.ax.plot(sol_x[i:i+2], sol_y[i:i+2], '-')
				self.ax.arrow(sol_x[i], sol_y[i], 0.3*cos(sol_yaw[i]), 0.3*sin(sol_yaw[i]), width=0.01)
				plt.pause(0.01)
			# plt.show()

	def set_constrains(self):
		raise NotImplementError

	def set_optimize_function(self):
		# Goal Cost
		goal_pos_cost = wg * sumsqr(self.pos[:, -1] - self.trajectory_desired[:, -1])
		goal_yaw_cost = wg_yaw * sumsqr(self.yaw[-1] - self.orientation_goal[2])
		goal_cost = goal_pos_cost + goal_yaw_cost

		# States Cost
		states_goal_cost =  ws * sumsqr(self.pos[:, 0:-1] - self.trajectory_desired[:, 0:-1]) 
		states_collision_cost = 0 # not necessary
		states_cost = states_goal_cost + states_collision_cost

		# Control Cost:
		control_min_cost = 0
		control_continuous_cost = 0
		for i in range(horizon):
			control_min_cost += self.U[:, i].T @ Q_u @ self.U[:, i]
			if i != 0:
				control_diff = self.U[:, i] - self.U[:, i-1]
				control_continuous_cost += control_diff.T @ Q_ud @ control_diff

		control_cost = control_min_cost + control_continuous_cost

		self.opti.minimize(
						   goal_cost 
						 + states_cost
			       		 + control_cost
			       			)


if __name__ == "__main__":
	p = nmpc_planner()
