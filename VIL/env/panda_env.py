import os
import time
import mujoco_py
import numpy as np
import gym
from gym import logger, spaces
# sys.path.append("/home/akhil/PhD/RoL/mujoco_panda-master")
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.controllers.torque_based_controllers import HuangVIC
from mujoco_panda.controllers.torque_based_controllers import VIC
from mujoco_panda.utils import VIC_func as func
from mujoco_panda.controllers.torque_based_controllers import VIC_config as cfg
import time
import random
import quaternion
import matplotlib.pyplot as plt


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PandaTrajTrack(gym.Env):#(modified_gym.GoalEnv):#

    def __init__(self, position_as_action = False, controller = "VIC", reward_type= "dense", pert_type='none' \
                 , n_actions= 3,  log_dir = None, render=True):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        #self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                             #render=True, compensate_gravity=False, smooth_ft_sensor=True)
        self.robot = PandaArm(model_path=MODEL_PATH + 'franka_panda_no_gripper.xml',
                             render=render, compensate_gravity=False, smooth_ft_sensor=True)

        if mujoco_py.functions.mj_isPyramidal(self.robot.model):
            print("Type of friction cone is pyramidal")
        else:
            print("Type of friction cone is eliptical")
        # self.init_jpos = np.array([-0.0242187,-0.26637015,-0.23036408,-1.99276073,-0.05414588,1.72812007, 0.52766157])
        #self.init_jpos = np.array([0., -0.7, 0, -2.356, 0, 1.656, 0.785])
        self.init_jpos = np.array([-2.03e-03, -8.42e-01, 1.09e-03, -2.26e+00, 5.80e-04, 1.41e+00, 8.24e-01])
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        time.sleep(1.0)
        self.render_robot = False
        self.position_as_action  = position_as_action
        self.reward_type = reward_type
        ''''
        Choose a controller type from VIC_Huang and VIC.
        VIC: the standard variable impedance controller where the delta K values are fed as actions
        VIC_Huang: VIC controller with an adaptive law, gamma parameter of the adaptive law is fed as action
        '''

        if controller == "VIC_Huang":
            self.controller = HuangVIC(self.robot, )
            if position_as_action:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER, cfg.DELTA_Z_LOWER]), \
                                           high=np.array([cfg.GAMMA_K_UPPER, cfg.DELTA_Z_UPPER]))
            else:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER]), \
                                     high=np.array([cfg.GAMMA_K_UPPER]))
        elif controller == "VIC":
            print("VIC controller")
            self.controller = VIC(self.robot, )
            if position_as_action:
                self.action_space = spaces.Box(low=np.array([cfg.DELTA_K_LOWER, cfg.DELTA_Z_LOWER]), \
                                           high=np.array([cfg.DELTA_K_UPPER, cfg.DELTA_Z_UPPER]))
            else:
                self.action_space = spaces.Box(low=np.array([cfg.DELTA_K_LOWER, cfg.DELTA_K_LOWER, cfg.DELTA_K_LOWER]), \
                                     high=np.array([cfg.DELTA_K_UPPER, cfg.DELTA_K_UPPER, cfg.DELTA_K_UPPER]))

        obs = self.get_obs()
        low = np.full(obs.shape, -float("inf"), dtype=np.float32)
        high = np.full(obs.shape, float("inf"), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=obs.dtype)
        #self.observation_space = spaces.Box(
            #low=np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Kz, cfg.LOWER_Fz]), \
            #high=np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Kz, cfg.UPPER_Fz]))
        self.fixed_traj = False
        self.random_traj = True
        self.timestep = cfg.T
        self.x_d_ddot_max = 0
        self.done = False
        self.max_num_it = cfg.MAX_NUM_IT
        self.F_offset = np.zeros(6)
        self.force_action= self.controller.cartesian_force
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = -cfg.Fd
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = cfg.Fd
        self.goal_ori = np.asarray(self.robot.ee_pose()[1])
        #self.goal_ori = np.array([0, -1, -3.82e-01,  0])
        if self.fixed_traj:
            self.traj_ddot, self.traj_dot, self.traj = func.generate_desired_trajectory_tc(\
                                                        self.robot, self.max_num_it+1, cfg.T,move_in_x=True)
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.i = 0
        self.x_d = np.asarray(self.robot.ee_pose()[0].copy())
        self.x_d_dot = np.zeros(6)
        self.x_d_ddot = np.zeros(6)
        self.obs_dict = self.controller.state_dict.copy()
        self.reset()
    # activate controller (simulation step and controller thread now running)

#design teh
    def robot_acceleration(self, prev_vel):
        x_ddot = (self.obs_dict['vel'][0:3].copy() - prev_vel)/ self.timestep
        self.obs_dict['acceleration'] = x_ddot
        return x_ddot


    def force_mean_filter(self, filter_type="mean", window_size= 5):
        if self.i < window_size:
            return (np.sum(self.F_history, axis=0)/(self.i+1))
        else:
            return (np.sum(self.F_history[self.i-window_size+1:,:], axis=0)/window_size)

    def set_render(self, rend):
        self.render_robot = rend

    def get_extra_obs(self):
        return self.obs_dict.copy()

    def get_ext_force(self):
        return (self.controller.virtual_ext_force.copy()[0:3])

    def get_external_states(self):
        return (self.controller.virtual_ext_force.copy()[0:3])

    def get_obs(self):
        self.obs_dict = self.controller.state_dict.copy()
        self.obs_dict['FT'] += self.controller.virtual_ext_force.copy()
        self.obs_dict['ext_force'] = self.controller.virtual_ext_force.copy()
        if self.stiffness_adaptation:
            #state = np.abs(self.controller.virtual_ext_force.copy()[0:3])
            state = np.concatenate((self.x_d - np.array(self.obs_dict["pose"][0:3].copy()), \
                       np.array(self.obs_dict["vel"][0:3].copy()), self.controller.virtual_ext_force.copy()[0:3]))
        else:
            state = np.concatenate((self.x_d - np.array(self.obs_dict["pose"][0:3].copy()), \
                                    np.array(self.obs_dict["vel"][0:3].copy()),))  # self.obs_dict['FT'][0:3].copy()))
        return state

    def get_reward(self, achieved_goal, desired_goal, input, cartesian_force, x_ddot):

        if np.sum(np.abs(self.controller.virtual_ext_force)) == 0:
            reward = -2*np.linalg.norm(input)/ np.sqrt(3)
        else:
            reward = np.clip(-np.linalg.norm(100 * (desired_goal - achieved_goal)), -1000, -0.5)
            reward -= np.linalg.norm(input)/ np.sqrt(3)
        if self.stiffness_adaptation:
            r_k = -np.linalg.norm(input)/ np.sqrt(3)#-np.mean(np.square(input))
        else:
            r_k = -np.mean(input)
        return reward

    def get_reward_basic(self, achieved_goal, desired_goal, input):
        obs_cost = np.sum(np.square(100*(desired_goal - achieved_goal)))
        act_cost = 0*0.1 * np.sum(np.square(input))
        # reward = torch.exp(-torch.sum((10*next_obs[:,0:3] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
        reward = -(obs_cost + act_cost)
        return reward

    #for HER
    def her_reward(self, achieved_goal, desired_goal, input):
        #r_track = np.exp(-goal_distance(30*achieved_goal, 30*desired_goal)) -1
        r_track = -goal_distance(20 * achieved_goal, 20 * desired_goal)
        #r_k =  np.exp(-np.linalg.norm(4*(input+1), axis = -1)) -1
        #r_k = -np.linalg.norm(input+1, axis = -1) / np.linalg.norm(np.array([2, 2, 2]))
        r_k = -np.mean(np.square(input), axis = -1)
        #print("test: ", input, input+1)
        reward = r_track#1*r_k
        #print(np.sum(r_track))
        #print("rewards : ", reward)
        return reward
    def change_goal(self):
        if self.fixed_traj or self.random_traj:
            obs_dict = self.controller.state_dict.copy()
            current_pose = obs_dict['pose'][0:3]
            current_vel = obs_dict['vel'][0:3]
            self.x_d_ddot = self.traj_ddot[:, self.i+1]
            self.x_d_dot  = self.traj_dot[:, self.i+1]
            self.x_d      = self.traj[:, self.i+1] #self.traj[:, -1]#
            self.x_d_dot[0:3] = (self.x_d - current_pose)/self.timestep
            self.x_d_ddot[0:3] = (self.x_d_dot[0:3] - current_vel) / self.timestep
            #print(self.x_d_dot[0:3], self.x_d_ddot[0:3] )
        else:
            obs_dict = self.controller.state_dict.copy()
            current_pose = obs_dict['pose'][0:3]
            current_vel = obs_dict['vel'][0:3]
            #self.x_d_ddot = np.concatenate((np.random.uniform(low=-.1, high=.1, size=(3,)), np.zeros(3)))
            self.x_d_ddot = self.x_d_ddot_max*np.sin(2*np.pi*self.max_num_it/(self.i+1))

            #print(self.x_d_ddot )
            self.x_d_dot += self.x_d_ddot * self.timestep
            print(self.x_d_dot)
            self.x_d     += self.x_d_dot[0:3] * self.timestep
            self.x_d_dot[0:3] = (self.x_d - current_pose) / self.timestep
            self.x_d_ddot[0:3] = (self.x_d_dot[0:3] - current_vel) / self.timestep


    def get_goal(self):
        return self.x_d.copy(), np.diag(np.array(self.obs_dict['K']))[0:3]

    def step(self, action):
        #print(action)
        #action = np.ones(3)
        if not self.stiffness_adaptation:
            action =  3000*np.square(action)
        #print(action)
        x_dot = self.obs_dict['vel'][0:3].copy()
        #print(action)
        for k in range(10):
            self.controller.set_goal(1*action, self.x_d, self.goal_ori, 0*self.x_d_dot, \
                                 0*self.x_d_ddot, goal_force=self.f_d[:, self.i])
            self.controller._send_cmd()
            self.robot.render()
            #self.controller.get_robot_states()
            #self.change_goal()

        if not self.stiffness_adaptation:
            if (self.i) % 1 == 0:
                if np.random.rand() >= 0:
                    self.controller.virtual_ext_force = 1*np.concatenate(( \
                            np.random.uniform(low=-10, high=10, size=(3,)), np.zeros(3)))
                else:
                    self.controller.virtual_ext_force = np.zeros(6)
            #print(self.controller.virtual_ext_force )
        self.controller.timestep += 1
        last_goal = self.x_d.copy()
        self.i += 1
        self.change_goal()
        obs = self.get_obs()
        x_ddot = self.robot_acceleration(x_dot)
        pose = self.obs_dict['pose'][0:3].copy()
        if (self.i > self.max_num_it-2):
            #print("doneeeeeee")
            done = True#(self.iteration >= self.max_num_it)
        else:
            done = False
        cartesian_force = self.obs_dict['cartesian_force'][0:3].copy()
        reward = self.get_reward_basic(last_goal, pose, action )
        info = {}
        if self.render_robot:
            self.robot.render()
        #print(obs[0:3])
        return obs, reward, done, info

    def reset(self):
        #print("resetting envs")
        index = 0  # np.random.randint(0, (0.9 * self.max_num_it))
        self.i = index
        #for i in range (100):
            #self.robot.render()
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        self.controller.reset()
        self.obs_dict = self.controller.state_dict.copy()
        if self.fixed_traj:
            self.traj_ddot, self.traj_dot, self.traj = func.generate_desired_trajectory_tc(\
                                                        self.robot, self.max_num_it+1,cfg.T,move_in_x=True)
            self.change_goal()
        elif self.random_traj:
            self.traj_ddot, self.traj_dot, self.traj = func.generate_random_trajectory(\
                                self.robot, self.max_num_it+1,cfg.T, x= True, y= True, z=True)
            self.change_goal()
        else:
            self.x_d = self.obs_dict['pose'][0:3].copy()
            self.x_d_dot = np.zeros(6)
            self.x_d_ddot_max = np.concatenate((np.random.uniform(low=-.1, high=.1, size=(3,)), np.zeros(3)))
            self.change_goal()
        #
        return self.get_obs()

    def render(self, mode=None, **kwargs):
        self.robot.render()


def make_panda_env(env_cfg):
    #num_it = env_cfg.overrides.get("trial_length", cfg.MAX_NUM_IT)
    controller = env_cfg.overrides.get("controller", "VIC")
    #control_rate = env_cfg.overrides.get("control_rate", cfg.PUBLISH_RATE)

    render = env_cfg.get("render", True)

    env = PandaTrajTrack(controller=controller,  render=render)

    return env


if __name__ == "__main__":

    VIC_env = PandaTrajTrack(controller = "VIC") #
    #gym.make("gym_robotic_ultrasound:ultrasound-v0")
    curr_ee, curr_ori = VIC_env.robot.ee_pose()
    print(VIC_env.robot.ee_pose()[1])
    # --------------------------------------

    VIC_env.controller.set_active(True)
    now_r = time.time()
    i = 0
    count = 0
    Data = []
    VIC_env.reset()
    #VIC_env.robot.render = True
    VIC_env.robot.render()
    print(i)
    while True:
        # get current robot end-effector pose
        timestep = VIC_env.controller.timestep / 1
        #print("helloooo",i, timestep)
        robot_pos, robot_ori = VIC_env.robot.ee_pose()
        render_frame(VIC_env.robot.viewer, robot_pos, robot_ori)
        render_frame(VIC_env.robot.viewer, VIC_env.x_d[:, i], VIC_env.goal_ori, alpha=0.2)
        if timestep >= i:
            elapsed_r = time.time() - now_r
            # render controller target and current ee pose using frames
            action = np.array([0,0,0])# 10*(1-2*np.random.random())#0#.000001#np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s,r,_,_ = VIC_env.step(action)
            print("reward: ", r)
            #print(s)
            i += 1
            if i == 4999:
                break
        VIC_env.robot.render()  # render the visualisation

    plt.plot(VIC_env.controller.F_history[:,2])
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()