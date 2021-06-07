from nmpc_planner import *

delta = 0.5 # this parameter is differnet

class nmpc_manuel(nmpc_planner):
	def __init__(self):
		nmpc_planner.__init__(self)
		


	def set_constrains(self):
		self.opti.subject_to(self.X[0:3, 0] == self.pos_start)
		self.opti.subject_to(self.X[6:9, 0] == self.orientation_start)
		for i in range(horizon):
			# dyanmics constain:
			self.opti.subject_to(self.X[:, i+1] == self.dynamics_update_function(self.X[:, i], self.U[:, i]))
			self.opti.subject_to(self.T[i] == g/(cos(self.roll[i])*cos(self.pitch[i]))) # no vertical net force

			# control constain:
			self.opti.subject_to(self.U[0, i] <= T_max)
			self.opti.subject_to(self.U[0, i] >= 0)
			self.opti.subject_to(self.U[1, i] <= angle_max)
			self.opti.subject_to(self.U[1, i] >= -angle_max)
			self.opti.subject_to(self.U[2, i] <= angle_max)
			self.opti.subject_to(self.U[2, i] >= -angle_max)
			self.opti.subject_to(self.U[3, i] <= yaw_rate_max)
			self.opti.subject_to(self.U[3, i] >= -yaw_rate_max)

			# ENV constrain:
			self.opti.subject_to(self.X[0, i+1] > ENV_X_MIN)
			self.opti.subject_to(self.X[0, i+1] < ENV_X_MAX)
			self.opti.subject_to(self.X[1, i+1] > ENV_Y_MIN)
			self.opti.subject_to(self.X[1, i+1] < ENV_Y_MAX)

			# Obstacle constain:
			delta_i = delta/len(static_obstacle)
			# delta_i = 0.3
			for obs_pos, size, uncertainty in zip(static_obstacle, static_obstacle_size, static_obstacle_uncertainty):
				current_pos = self.pos[:, i]	# current position

				lhs = 0
				for n in range(2):
					c = erfinv(1-2*delta_i) * np.sqrt(2*1*uncertainty[n]*1)
					d = size[n]
					diff = current_pos[n] - obs_pos[n]
					lhs += (diff/(d+c))**2

				self.opti.subject_to(lhs >= 2)


if __name__ == "__main__":
	start = [0, 0, 0, 0, 0, np.pi/2]
	goal = [10, 6, 0, 0, 0, 0]

	p = nmpc_manuel()
	plot_stats = False
	num_iter = 100
	for i in range(num_iter):
		print(start)
		print(goal)
		p.generate_trajectory(start, goal, plot=True)
		solution = p.solution
		inputs_sol = p.inputs_sol
		start = [solution[0, -1], solution[1, -1],  solution[2, -1], 0, 0, solution[8, -1]]

		# plot for states
		if plot_stats:
			plt.figure()
			plt.subplot(3, 3, 1)
			sol_x = solution[0, :]
			plt.plot(range(len(sol_x)), sol_x)
			plt.title("x vs. Horizon (m)")

			plt.subplot(3, 3, 2)
			sol_y = solution[1, :]
			plt.plot(range(len(sol_y)), sol_y)
			plt.title("y vs. Horizon (m)")

			plt.subplot(3, 3, 3)
			sol_z = solution[2, :]
			plt.plot(range(len(sol_z)), sol_z)
			plt.title("z vs. Horizon (m)")

			plt.subplot(3, 3, 4)
			sol_vx = solution[3, :]
			plt.plot(range(len(sol_vx)), sol_vx)
			plt.title("vx vs. Horizon (m/s)")

			plt.subplot(3, 3, 5)
			sol_vy = solution[4, :]
			plt.plot(range(len(sol_vy)), sol_vy)
			plt.title("vy vs. Horizon (m/s)")

			plt.subplot(3, 3, 6)
			sol_vz = solution[5, :]
			plt.plot(range(len(sol_vz)), sol_vz)
			plt.title("vz vs. Horizon (m/s)")

			plt.subplot(3, 3, 7)
			sol_roll = solution[6, :]
			plt.plot(range(len(sol_roll)), sol_roll * 180/np.pi)
			plt.title("roll vs. Horizon (degree)")

			plt.subplot(3, 3, 8)
			sol_pitch = solution[7, :]
			plt.plot(range(len(sol_pitch)), sol_pitch * 180/np.pi)
			plt.title("pitch vs. Horizon (degree)")

			plt.subplot(3, 3, 9)
			sol_yaw = solution[8, :]
			plt.plot(range(len(sol_yaw)), sol_yaw * 180/np.pi)
			plt.title("yaw vs. Horizon (degree)")

			# Plot for control
			plt.figure()
			plt.subplot(2, 2, 1)
			sol_T = inputs_sol[0, :]
			plt.plot(range(len(sol_T)), sol_T*m)
			plt.title("Thrust force vs. Horizon (Newton)")

			plt.subplot(2, 2, 2)
			sol_rolld = inputs_sol[1, :]
			plt.plot(range(len(sol_rolld)), sol_rolld*180/np.pi)
			plt.title("Roll_d vs. Horizon (degree)")

			plt.subplot(2, 2, 3)
			sol_pitchd = inputs_sol[2, :]
			plt.plot(range(len(sol_pitchd)), sol_pitchd*180/np.pi)
			plt.title("pitch_d vs. Horizon (degree)")

			plt.subplot(2, 2, 4)
			sol_yaw_rated = inputs_sol[3, :]
			plt.plot(range(len(sol_yaw_rated)), sol_yaw_rated*180/np.pi)
			plt.title("yaw_rate_d vs. Horizon (degree)")

			plt.show()


		# distance_s_g = np.linalg.norm(np.array(start[0:3]) - np.array(goal[0:3]))
		# if distance_s_g >= 0.5:
		# 	continue

		available = False
		while not available:
			goal_x = np.random.uniform(ENV_X_MIN, ENV_X_MAX)
			goal_y = np.random.uniform(ENV_Y_MIN, ENV_Y_MAX)
			goal_yaw = np.random.uniform(-np.pi, np.pi)
			count = 0
			for obs_pos, size in zip(static_obstacle, static_obstacle_size):
				distance = np.linalg.norm(np.array(obs_pos[0:2]) - np.array([goal_x, goal_y]))
				if distance <= 1.0 * np.sqrt(2) * np.max(size):
					break
				else:
					count += 1
					if count == len(static_obstacle):
						available = True

		goal = [goal_x, goal_y, 0, 0, 0, goal_yaw]

		plt.waitforbuttonpress()
