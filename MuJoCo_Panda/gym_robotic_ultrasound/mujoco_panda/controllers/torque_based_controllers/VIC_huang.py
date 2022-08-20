from mujoco_panda.utils.tf import quatdiff_in_euler
from .configs import BASIC_HYB_CONFIG
#import VIC_config as cfg
from mujoco_panda.controllers.torque_based_controllers.VIC_base_controller import BaseControllerVIC
from mujoco_panda.controllers.torque_based_controllers import VIC_Huang_config as cfg
import numpy as np
import quaternion
import time

class HuangVIC(BaseControllerVIC):
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
        super().__init__(robot_object)
        self.demo_data_dict = {}
        self.demo_data = []
        self.gamma = np.identity(18)
        self.gamma_K = cfg.GAMMA_K_INIT
        self.gamma[14, 14] = self.gamma_K
        self.lam = np.zeros(18)
        self.Kv = cfg.K_v
        self.P = cfg.P
        self.B_hat_lower = cfg.B_hat_lower
        self.B_hat_upper = cfg.B_hat_upper
        self.K_hat_lower = cfg.K_hat_lower
        self.K_hat_upper = cfg.K_hat_upper
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)

    def _compute_cmd(self):
        """
        Actual computation of command given the desired goal states

        :return: computed joint torque values
        :rtype: np.ndarray (7,)
        """
        #self.gamma[8, 8] = self.action[0]  # gamma B
        #print("helloooo")
        self.gamma[14, 14] = self.action  # gamma K

        x, x_dot, delta_x, jacobian, robot_inertia, F_ext_2D = \
            self.fetch_states(self.timestep,  self.goal_pos)

        xi = self.get_xi(x_dot, self.goal_vel, self.goal_acc, delta_x, \
                         self.x_dot_history, self.timestep, 1/self.control_rate)
        #print(self.goal_force)
        self.lam = self.lam.reshape([18, 1]) + self.get_lambda_dot(self.gamma, xi, self.Kv, self.P, \
                    self.goal_force, F_ext_2D, self.timestep,).reshape([18, 1])
        #self.lam[14]= np.clip(self.lam[14], -2000, 8000)
        #print(self.lam[14])
        M_hat, B_hat, K_hat = self.update_MBK_hat(self.lam, self.M, self.B, self.K)
        #print(B_hat, K_hat)
        #print(np.shape(B_hat), np.shape(K_hat))#, B_hat[2,2],K_hat[2,2])
        self.perform_torque_Huang1992(M_hat, B_hat, K_hat, self.goal_acc, self.goal_vel, x, \
                            x_dot, self.goal_pos,  F_ext_2D, jacobian, robot_inertia)


    def reset(self):
        super().reset()
        self.gamma = np.identity(18)
        self.gamma[14, 14] = self.gamma_K
        self.lam = np.zeros(18)
        self.gamma_B_hist = np.zeros(self.max_num_it)
        self.gamma_K_hist = np.zeros(self.max_num_it)

    def get_lambda_dot(self, gamma, xi, K_v, P, F_d, F_ext_2D, i, ):
        T = 1/self.control_rate#float(time_per_iteration[i] - time_per_iteration[i - 1])
        #print(F_ext_2D - F_d.reshape([6, 1]))
        return np.linalg.multi_dot \
                ([-np.linalg.inv(gamma), xi.T, np.linalg.inv(K_v), P, F_ext_2D - F_d.reshape([6, 1])]) * T

    def update_MBK_hat(self, lam, M, B, K):
        M_hat = self.M  # + np.diagflat(lam[0:6]) M is chosen to be constant
        K_hat = self.K.copy() + np.diagflat(lam[12:18])
        K_hat = np.clip(K_hat, self.K_hat_lower, self.K_hat_upper)
        B_hat = B.copy()
        #K_hat[2,2] = 100
        B_hat[2,2] = 2*np.sqrt(K_hat[2,2])
        B_hat = np.clip(B_hat, self.B_hat_lower, self.B_hat_upper)
        #self.K = K_hat
        #self.B = B_hat
        print(np.diag(K_hat)[:3])
        return M_hat, B_hat, K_hat

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
        self.demo_data_dict["goal_force"] = self.goal_forc
