import os
import time
import pathlib
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
from mujoco_panda.controllers.torque_based_controllers import VIC_Huang_config as cfg
import time
import random
import quaternion
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class VIChuangEnv(gym.Env):

    def __init__(self, position_as_action = False, controller = "VIC_Huang", control_rate = None, \
                 log_dir = None, max_num_it = cfg.MAX_NUM_IT, render=True):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        #self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                             #render=True, compensate_gravity=False, smooth_ft_sensor=True)
        self.robot = PandaArm(model_path=MODEL_PATH + 'panda_ultrasound.xml',
                             render=render, compensate_gravity=False, smooth_ft_sensor=False)
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
        self.position_as_action  = position_as_action

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
        else:
            raise ValueError("Invalid contorller type")
        self.stiffness_adaptation   = True
        self.controller.adaptation = self.stiffness_adaptation
        self.fixed_traj = True
        self.random_traj = False
        self.timestep = cfg.T
        self.done = False
        self.max_num_it = self.controller.max_num_it
        self.F_offset = np.zeros(6)

        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = -cfg.Fd
        # set desired pose/force trajectory
        self.f_d = func.generate_Fd_steep(self.max_num_it, cfg.Fd, cfg.T)
        self.f_d[2, :] = cfg.Fd
        self.f_d = np.array([0,0,195, 0,0,0])
        self.goal_ori = np.asarray(self.robot.ee_pose()[1])
        #self.goal_ori = np.array([0, -1, -3.82e-01,  0])
        if self.fixed_traj:
            self.traj_ddot, self.traj_dot, self.traj = func.generate_desired_trajectory_tc( \
                self.robot, self.max_num_it + 1, cfg.T, move_in_x=True)
        #plt.plot(self.traj[0, :])
        #plt.show()
        self.i = 0
        self.x_d = np.zeros(3)
        self.pose = self.x_d.copy()
        self.x_d_dot = np.zeros(6)
        self.x_d_ddot = np.zeros(6)
        self.obs_dict = self.controller.state_dict.copy()
        obs = self.get_obs()
        low = np.full(obs.shape, -float("inf"), dtype=np.float32)
        high = np.full(obs.shape, float("inf"), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=obs.dtype)
        # self.observation_space = spaces.Dict(dict(
        # observation=spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32'),
        # ))
        self.reset()
    # activate controller (simulation step and controller thread now running)

    def robot_acceleration(self, prev_vel):
        x_ddot = (self.obs_dict['vel'][0:3].copy() - prev_vel)/ self.timestep
        self.obs_dict['acceleration'] = x_ddot
        return x_ddot
    #design teh
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
        return (self.obs_dict['FT'][0:3].copy())

    def get_obs(self):
        self.obs_dict = self.controller.state_dict.copy()
        state = np.concatenate ((np.array(self.obs_dict["FT"][0:3]),self.x_d - np.array(self.obs_dict["pose"][0:3].copy()), \
                         np.array(self.obs_dict["vel"][0:3].copy())))
        return state
    def get_reward(self, pose_goal, force_goal, pose,force, input):
        pose_cost = np.sum(np.square(100 * (pose_goal - pose)))
        force_cost = np.sum(np.square(1 * (force_goal[2] - force[2])))
        act_cost = 0 * 0.1 * np.sum(np.square(input))
        reward = -(1*pose_cost + 1*force_cost + 0*act_cost)
        return reward

    def change_goal(self):
        if self.fixed_traj or self.random_traj:
            obs_dict = self.controller.state_dict.copy()
            current_pose = obs_dict['pose'][0:3]
            current_vel = obs_dict['vel'][0:3]
            #self.x_d_ddot = self.traj_ddot[:, self.i+1]
            #self.x_d_dot  = self.traj_dot[:, self.i+1]
            self.x_d = self.traj[:, 0]#self.traj[:, self.i+1] #self.traj[:, -1]#

            self.x_d_dot[0:3] = (self.x_d - current_pose)/self.timestep/10
            self.x_d_ddot[0:3] = (self.x_d_dot[0:3] - current_vel) / self.timestep/10
            #print(self.x_d_dot[0:3], self.x_d_ddot[0:3] )

    def get_goal(self):
        return self.x_d.copy(), np.diag(np.array(self.obs_dict['K']))[0:3]

    def step(self, action):
        #if self.position_as_action:
        if not self.stiffness_adaptation:
            action =  1*np.square(action)
        #action = np.array([5, 5, -0.0])
        x_dot = self.obs_dict['vel'][0:3].copy()
        #action = np.array([5, 5, 10])
        for k in range(1):
            self.controller.set_goal(action, self.x_d, self.goal_ori, 1*self.x_d_dot, \
                                 1*self.x_d_ddot, goal_force=self.f_d)
            self.controller._send_cmd()
            #self.controller.get_robot_states()
            #self.change_goal()
        #self.controller.timestep += 1
        last_pose_goal = self.x_d.copy()[0:2]
        last_force_goal = self.f_d.copy()
        self.i += 1
        self.change_goal()
        obs = self.get_obs()
        x_ddot = self.robot_acceleration(x_dot)
        pose = self.obs_dict['pose'][0:2].copy()
        force = self.obs_dict['FT'][0:3].copy()
        print(force)
        if (self.i >= self.max_num_it-1)  or (np.max(np.abs(force)) > 500 or  (np.sum(np.abs(self.obs_dict['pose'][3:]))) > 0.1\
                        or (np.sum(np.abs(self.obs_dict['pose'][0:3] - self.x_d))) > 0.1):
            done = True#(self.iteration >= self.max_num_it)
            #print("done")
        else:
            done = False
        reward = self.get_reward(last_pose_goal, last_force_goal, pose,force, action )
        info = {}
        self.robot.render()
        return obs, reward, done, info

    def reset(self):
        #print("resetting envs")
        index = 0  # np.random.randint(0, (0.9 * self.max_num_it))
        self.i = index
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        self.controller.reset()
        return self.get_obs()

    def render(self, **kwargs):
        self.robot.render()


def make_vic_huang_env(env_cfg):

    #num_it = env_cfg.overrides.get("trial_length", cfg.MAX_NUM_IT)
    controller = env_cfg.overrides.get("controller", "VIC_Huang")
    #control_rate = env_cfg.overrides.get("control_rate", cfg.PUBLISH_RATE)
    
    render = env_cfg.get("render", True)
    
    env = VIChuangEnv(controller=controller,  render=render)

    return env

if __name__ == "__main__":

    VIC_env = VIChuangEnv(controller = "VIC_Huang")
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
    for i in range(1):
        VIC_env.robot.render()
        print(i)
    while True:
        # get current robot end-effector pose
        timestep = VIC_env.controller.timestep / 1
        #print("helloooo",i, timestep)
        robot_pos, robot_ori = VIC_env.robot.ee_pose()
        render_frame(VIC_env.robot.viewer, robot_pos, robot_ori)
        render_frame(VIC_env.robot.viewer, VIC_env.x_d, VIC_env.goal_ori, alpha=0.2)
        if True:
            elapsed_r = time.time() - now_r
            # render controller target and current ee pose using frames
            action = 0.00001#np.array([5,5,-0.0])# 10*(1-2*np.random.random())#0#.000001#np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s,r,_,_ = VIC_env.step(action)
            #print("reward: ", r)
            #print(i)
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