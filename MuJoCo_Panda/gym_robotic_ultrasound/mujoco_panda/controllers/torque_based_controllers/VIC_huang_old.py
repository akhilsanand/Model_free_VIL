from mujoco_panda.utils.tf import quatdiff_in_euler
from .configs import BASIC_HYB_CONFIG
#import VIC_config as cfg
from mujoco_panda.controllers.torque_based_controllers import VIC_config as cfg
import numpy as np
import quaternion
import time

class HuangVIC():
    """
    Torque-based task-space hybrid force motion controller.
    Computes the joint torques required for achieving a desired
    end-effector pose and/or wrench. Goal values and directions
    are defined in cartesian coordinates.

    First computes cartesian force for achieving the goal using PD
    control law, then computes the corresponding joint torques using 
    :math:`\tau = J^T F`.
    
    """

    def __init__(self, robot_object,  control_rate=None, *args, **kwargs):
        """
        contstructor

        :param robot_object: the :py:class:`PandaArm` object to be controlled
        :type robot_object: PandaArm
        :param config: dictionary of controller parameters, defaults to 
            BASIC_HYB_CONFIG (see config for reference)
        :type config: dict, optional
        """
        self._robot = robot_object
        #super(VIC,self).__init__(robot_object, config)
        self.t_prev = time.time()
        self.state_dict = {}
        self.demo_data_dict = {}
        self.demo_data = []
        self.new_goal = True
        self.goal_pos, self.goal_ori = self._robot.ee_pose()
        self.goal_vel = np.zeros(6)
        self.goal_acc = np.zeros(6)
        self.goal_force = np.zeros(6)
        self.control_rate = cfg.PUBLISH_RATE
        self.M = cfg.M
        self.K = cfg.K
        self.B = cfg.B
        self.max_num_it = cfg.MAX_NUM_IT
        self.gamma = np.identity(18)
        self.gamma_K = cfg.GAMMA_K_INIT
        self.gamma[14, 14] = self.gamma_K
        self.lam = np.zeros(18)
        self.K_full = np.zeros((6,6))
        self.Kv = cfg.K_v
        self.P = cfg.P
        self.B_hat_lower = cfg.B_hat_lower
        self.B_hat_upper = cfg.B_hat_upper
        self.K_hat_lower = cfg.K_hat_lower
        self.K_hat_upper = cfg.K_hat_upper
        self.timestep = 0
        self.F_history = np.zeros((self.max_num_it, 6))

        self.x_history = np.zeros((6, self.max_num_it))
        self.x_dot_history = np.zeros((6, self.max_num_it))
        self.p_hist = np.zeros((3, self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6, self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist = np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)

        self._use_null_ctrl = True

        if self._use_null_ctrl:
            self._null_Kp = np.diag([5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0])
            self._null_ctrl_wt = 2.5
        self._pos_threshold = 0.025
        self._angular_threshold =  0.01
        self._cmd = self._robot.sim.data.ctrl[self._robot.actuated_arm_joints].copy()
        self.get_robot_states()

    def set_active(self, status=True):
        """
        Override parent method to reset goal values

        :param status: To deactivate controller, set False. Defaults to True.
        :type status: bool, optional
        """
        if status:
            self.goal_pos, self.goal_ori = self._robot.ee_pose()
            self.goal_vel = np.zeros(6)
            self.goal_acc = np.zeros(6)
            self.goal_force = np.zeros(6)
        self._is_active = status

    def _compute_cmd(self):
        """
        Actual computation of command given the desired goal states

        :return: computed joint torque values
        :rtype: np.ndarray (7,)
        """
        #self.gamma[8, 8] = self.action[0]  # gamma B
        self.gamma[14, 14] = self.action  # gamma K

        x, x_dot, delta_x, jacobian, robot_inertia, F_ext_2D = \
            self.fetch_states(self.timestep,  self.goal_pos)


        xi = self.get_xi(x_dot, self.goal_vel, self.goal_acc, delta_x, \
                         self.x_dot_history, self.timestep, 1/self.control_rate)

        self.lam = self.lam.reshape([18, 1]) + self.get_lambda_dot(self.gamma, xi, self.Kv, self.P, \
                    self.goal_force, F_ext_2D, self.timestep,).reshape([18, 1])
        M_hat, B_hat, K_hat = self.update_MBK_hat(self.lam, self.M, self.B, self.K)
        self.K_full = K_hat
        #print(B_hat, K_hat)
        #print(np.shape(B_hat), np.shape(K_hat))#, B_hat[2,2],K_hat[2,2])
        torque = self.perform_torque_Huang1992(M_hat, B_hat, K_hat, self.goal_acc, self.goal_vel, x, \
                            x_dot, self.goal_pos,  F_ext_2D, jacobian, robot_inertia)

        u = torque  # desired joint torque

        if np.any(np.isnan(u)):
            u = self._cmd
        else:
            self._cmd = u
        #print("torque ", torque)
        if self._use_null_ctrl: # null-space control, if required

            null_space_filter = self._null_Kp.dot(
                np.eye(7) - jacobian.T.dot(np.linalg.pinv(jacobian.T, rcond=1e-3)))
            # add null-space torque in the null-space projection of primary task
            self._cmd = self._cmd + null_space_filter.dot(
                    self._robot._neutral_pose-self._robot.joint_positions()[:7])
        self.timestep = self.timestep + 1
        #print("cmd: ", self._cmd)
        #print("ee_pose ", self._robot.ee_pose()[0])
        #print("smoothed FT reading: ", self._robot.get_ft_reading(pr=True))
        return self._cmd

    def _send_cmd(self):

        now_c = time.time()
        self._compute_cmd()
        self._robot.set_joint_commands(
                self._cmd, joints=self._robot.actuated_arm_joints, compensate_dynamics=False)
        self._robot.sim_step(render=False)
        elapsed_c = time.time() - now_c
        sleep_time_c = (1./self.control_rate) - elapsed_c
        if sleep_time_c > 0.0:
            time.sleep(sleep_time_c)
        self.get_robot_states()

    def stop_controller_cleanly(self):
        """
        Method to be called when stopping controller. Stops the controller thread and exits.
        """
        self._is_active = False
        #self._logger.info ("Stopping controller commands; removing ctrl values.")
        self._robot.set_joint_commands(np.zeros_like(self._robot.actuated_arm_joints),self._robot.actuated_arm_joints)
        self._robot._ignore_grav_comp=False
        #self._logger.info ("Stopping controller thread. WARNING: PandaArm->step() method has to be called separately to continue simulation.")
        self._is_running = False



    def set_goal(self, action, goal_pos, goal_ori=None, goal_vel=np.zeros(6), goal_acc=np.zeros(6), goal_force=None):
        """
        change the target for the controller
        """

        #self.timestep = timestep
        self.action = action
        self.goal_pos = goal_pos
        self.goal_ori = goal_ori
        self.goal_vel = goal_vel
        self.goal_acc = goal_acc
        self.goal_force = goal_force
        self.new_goal = True
        #print(self.goal_force)


    def reset(self):
        self.timestep = 0
        self.gamma = np.identity(18)
        self.gamma[14, 14] = self.gamma_K
        self.lam = np.zeros(18)
        self.goal_pos, self.goal_ori = self._robot.ee_pose()
        self.goal_vel = np.zeros(6)
        self.goal_acc = np.zeros(6)
        self.goal_force = np.zeros(6)
        self.F_history = np.zeros((self.max_num_it, 6))
        self.x_history = np.zeros((6, self.max_num_it))
        self.x_dot_history = np.zeros((6, self.max_num_it))
        self.p_hist = np.zeros((3, self.max_num_it))
        self.Fz_history = np.zeros(self.max_num_it)
        self.h_e_hist = np.zeros((6, self.max_num_it))
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)
        self.Kp_pos_hist = np.zeros(self.max_num_it)
        self.Kp_z_hist = np.zeros(self.max_num_it)
        self.Kd_z_hist = np.zeros(self.max_num_it)

    def perform_torque_Huang1992(self, M, B, K, x_d_ddot, x_d_dot, x, x_dot, p_d, F_ext_2D, jacobian, robot_inertia):
        self.demo_data_dict["k"] = K[2, 2]
        self.demo_data_dict["x_dot_delta"] = self.get_x_dot_delta(x_d_dot, x_dot)
        self.demo_data_dict["delta_x"] = self.get_delta_x(x, p_d, two_dim=True)
        self.demo_data_dict["W"] = self.get_W(jacobian, robot_inertia, inv=True)
        a = np.linalg.multi_dot([jacobian.T, self.get_W(jacobian, robot_inertia, inv=True), np.linalg.inv(M)])
        b = np.array([np.dot(M, x_d_ddot)]).reshape([6, 1]) + np.array(
            [np.dot(B, self.get_x_dot_delta(x_d_dot, x_dot))]).reshape([6, 1]) + np.array(
            [np.dot(K, 10*self.get_delta_x(x, p_d, two_dim=True))]).reshape([6, 1])
        #print(self.get_delta_x(x, p_d, two_dim=True))
        #c = 0*self.torque_compensation.reshape([7, 1]) # fix
        d = (np.identity(6) - np.dot(self.get_W(jacobian, robot_inertia, inv=True), np.linalg.inv(M))).reshape([6, 6])
        total_torque = np.array([np.dot(a, b)]).reshape([7, 1]) + np.array(
            [np.linalg.multi_dot([jacobian.T, d, F_ext_2D])]).reshape([7, 1])
        #self.demo_data_dict["x_dot_delta"] = self.get_x_dot_delta(x_d_dot, x_dot))])

        return total_torque.reshape(7,)

 #Not used in this impelenmnetatin , used for collecting demo data in the threading based version
    def set_demonstration_data(self):
        self.demo_data_dict["goal_acc"] = self.goal_acc
        self.demo_data_dict["goal_vel"] = self.goal_vel
        self.demo_data_dict["pose"] = self.get_x(self.goal_ori) # x
        self.demo_data_dict["x_dot"] = np.concatenate((self._robot.ee_velocity()[0], self._robot.ee_velocity()[1]))
        self.demo_data_dict["goal_pos"] = self.goal_pos
        self.demo_data_dict["F_ext_2D"] = np.array([0, 0, self._robot.get_ft_reading()[0][2], 0, 0, 0]).reshape([6, 1])
        self.demo_data_dict["J"] = self._robot.jacobian() # jacobian
        self.demo_data_dict["robot_inertia"] = self._robot.mass_matrix()
        self.demo_data_dict["FT"] = np.concatenate((self._robot.get_ft_reading()[0],self._robot.get_ft_reading()[1] ))
        self.demo_data_dict["goal_ori"] = self.goal_ori
        self.demo_data_dict["goal_force"] = self.goal_force

    def force_mean_filter(self, filter_type="mean", window_size= 5):
        if self.timestep < window_size:
            return (np.sum(self.F_history, axis=0)/(self.timestep+1))
        else:
            return (np.sum(self.F_history[self.timestep-window_size+1:,:], axis=0)/window_size)

    def get_robot_states(self):
        self.state_dict["pose"] = self.get_x(self.goal_ori) # x
        self.state_dict["J"] = self._robot.jacobian() # jacobian
        self.state_dict["FT_raw"] = np.concatenate((self._robot.get_ft_reading()[0],self._robot.get_ft_reading()[1]))
        self.F_history[self.timestep, :] = self.state_dict["FT_raw"]
        self.state_dict["FT"] = self.force_mean_filter()
        self.state_dict["vel"] = np.concatenate((self._robot.ee_velocity()[0],self._robot.ee_velocity()[1] ))
        self.state_dict["M"] = self._robot.mass_matrix()
        self.state_dict["K"] = self.K_full


    def fetch_states(self, i, p_d ):
        #self.get_robot_states()
        x = self.state_dict["pose"]
        self.x_history[:, i] = x
        #p = x[:3]
        jacobian = self.state_dict["J"]
        robot_inertia = self.state_dict["M"]
        #self.ee_force = self.sim.data.cfrc_ext[self.probe_id][-3:] #check hand_force term
        Fz = self.state_dict["FT"][2] #self._robot.get_ft_reading()[0][2]
        F_ext = np.array([0, 0, Fz, 0, 0, 0])
        F_ext_2D = F_ext.reshape([6, 1])
        if 0:  # correct
            x_dot = self.get_derivative_of_vector(self.x_history, i,0)
        else:
            #ee_vel, ee_omg = self._robot.ee_velocity()
            x_dot = self.state_dict["vel"]#np.concatenate((ee_vel, ee_omg))
        self.x_dot_history[:, i] = x_dot
        delta_x = self.get_delta_x(x, p_d)
        #Rot_e = self.ee_ori_mat
        return x, x_dot, delta_x, jacobian, robot_inertia,  F_ext_2D

    def get_lambda_dot(self, gamma, xi, K_v, P, F_d, F_ext_2D, i, ):

        T = 1/self.control_rate#float(time_per_iteration[i] - time_per_iteration[i - 1])
        return np.linalg.multi_dot \
                ([-np.linalg.inv(gamma), xi.T, np.linalg.inv(K_v), P, F_ext_2D - F_d.reshape([6, 1])]) * T

    def update_MBK_hat(self, lam, M, B, K):
        M_hat = M  # + np.diagflat(lam[0:6]) M is chosen to be constant
        K_hat = K + np.diagflat(lam[12:18])
        K_hat = np.clip(K_hat, self.K_hat_lower, self.K_hat_upper)
        B_hat = B.copy()
        B_hat[2,2] = np.sqrt(K_hat[2,2])
        # ensure_limits(1,5000,M_hat)
        B_hat = np.clip(B_hat, self.B_hat_lower, self.B_hat_upper)

        return M_hat, B_hat, K_hat

    def get_x_dot_delta(self,x_d_dot, x_dot, two_dim=True):
        if two_dim == True:
            return (x_d_dot - x_dot).reshape([6, 1])
        else:
            return x_d_dot - x_dot

    def get_x_ddot_delta(self,x_d_ddot, v_history, i, dt ):
        a = self.get_derivative_of_vector(v_history, i, dt)
        return x_d_ddot - a

    def get_xi(self, x_dot, x_d_dot, x_d_ddot, delta_x, x_dot_history, i, dt):
        E = -delta_x
        E_dot = -self.get_x_dot_delta(x_d_dot, x_dot, two_dim=False)
        E_ddot = -self.get_x_ddot_delta(x_d_ddot, x_dot_history, i, dt)
        E_diag = np.diagflat(E)
        E_dot_diag = np.diagflat(E_dot)
        E_ddot_diag = np.diagflat(E_ddot)
        return np.block([E_ddot_diag, E_dot_diag, E_diag])

    def get_derivative_of_vector(self, history, iteration, dt):
        size = history.shape[0]
        if iteration > 0:
            if dt > 0:
                return np.subtract(history[:, iteration], history[:, iteration - 1]) / dt
        return np.zeros(size)

    def quatdiff_in_euler_radians(self, quat_curr, quat_des):
        curr_mat = quaternion.as_rotation_matrix(quat_curr)
        des_mat = quaternion.as_rotation_matrix(quat_des)
        rel_mat = des_mat.T.dot(curr_mat)
        rel_quat = quaternion.from_rotation_matrix(rel_mat)
        vec = quaternion.as_float_array(rel_quat)[1:]
        if rel_quat.w < 0.0:
            vec = -vec
        return -des_mat.dot(vec)

    def get_x(self, goal_ori):
        pos_x, curr_ori = self._robot.ee_pose()
        ee_current_ori = quaternion.as_quat_array(curr_ori)
        goal_ori = quaternion.as_quat_array(goal_ori)

        rel_ori = self.quatdiff_in_euler_radians(ee_current_ori, goal_ori)  # used to be opposite  # used to be opposite
        return np.append(pos_x, rel_ori)

    def get_delta_x(self,x, p_d, two_dim=False):
        #print(self.goal_pos, p_d , x[:3])
        delta_pos = p_d - x[:3]
        delta_ori = 10*x[3:]   # check and change , hack for now,
        if two_dim == True:
            return np.array([np.append(delta_pos, delta_ori)]).reshape([6, 1])
        else:
            return np.append(delta_pos, delta_ori)

    def get_W(self, jacobian, robot_inertia, inv=False):
        W = np.linalg.multi_dot([jacobian, np.linalg.inv(robot_inertia), jacobian.T])
        if inv == True:
            return np.linalg.inv(W)
        else:
            return W

