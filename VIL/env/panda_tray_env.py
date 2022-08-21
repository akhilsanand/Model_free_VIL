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
from mujoco_panda.controllers.torque_based_controllers.VIC_env_configs import VIC_tray_config as cfg
import time
import random
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PandaTrayEnv(gym.Env):

    def __init__(self, position_as_action=False, controller="VIC", reward_type="dense", pert_type='none' \
                 , n_actions=3, log_dir=None, render=False, goal_type='random'):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        #self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                             #render=True, compensate_gravity=False, smooth_ft_sensor=True)
        self.robot = PandaArm(model_path=MODEL_PATH + 'panda_tray_multi_balls.xml',
                             render=False, compensate_gravity=True, \
                              grav_comp_model_path=MODEL_PATH + 'franka_panda_with_tray.xml', smooth_ft_sensor=True) #'franka_panda_with_tray.xml'
        if mujoco_py.functions.mj_isPyramidal(self.robot.model):
            print("Type of friction cone is pyramidal")
        else:
            print("Type of friction cone is eliptical")
        self.init_jpos = np.array([-2.03e-03, -8.42e-01, 1.09e-03, -2.26e+00, 5.80e-04, 1.41e+00, 8.24e-01 - np.pi/4,])
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        print(self.robot._sim.data.qpos)
        self.object1_init_pose = np.concatenate((self.robot.body_pose("ball1")[0],self.robot.body_pose("ball1")[1] ))
        self.object2_init_pose = np.concatenate((self.robot.body_pose("ball2")[0], self.robot.body_pose("ball2")[1]))
        self.object3_init_pose = np.concatenate((self.robot.body_pose("ball3")[0], self.robot.body_pose("ball3")[1]))
        self.object4_init_pose = np.concatenate((self.robot.body_pose("ball4")[0], self.robot.body_pose("ball4")[1]))
        self.render_robot = render
        self.position_as_action  = position_as_action
        self.reward_type = reward_type
        self.pert_type = pert_type
        self.n_actions = n_actions
        self.goal_type = goal_type
        self.timestep = cfg.T
        self.max_num_it = cfg.MAX_NUM_IT
        self.done = False
        self.goal_ori = np.asarray(self.robot.ee_pose()[1])
        self.i = 0
        self.x_d = np.asarray(self.robot.ee_pose()[0].copy())
        self.x_d_dot = np.zeros(6)
        self.x_d_ddot = np.zeros(6)
        self.action = None
        self.prev_action = None
        self.f_offset = self.robot.get_ft_reading()[0].copy()

        ''''
        Choose a controller type from VIC_Huang and VIC.
        VIC: the standard variable impedance controller where the delta K values are fed as actions
        VIC_Huang: VIC controller with an adaptive law, gamma parameter of the adaptive law is fed as action
        '''

        if controller == "VIC_Huang":
            self.controller = HuangVIC(self.robot, cfg)
            if position_as_action:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER, cfg.DELTA_Z_LOWER]), \
                                           high=np.array([cfg.GAMMA_K_UPPER, cfg.DELTA_Z_UPPER]))
            else:
                self.action_space = spaces.Box(low=np.array([cfg.GAMMA_K_LOWER]), \
                                     high=np.array([cfg.GAMMA_K_UPPER]))
        elif controller == "VIC":
            print("VIC controller")
            self.controller = VIC(self.robot, cfg)
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
        self.obs_dict = self.controller.state_dict.copy()
        self.reset()


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

    def get_ext_forces(self):
        return (np.array(self.obs_dict["FT_raw"][0:3].copy()))

    def get_external_states(self):
        return np.concatenate((self.x_d - np.array(self.obs_dict["pose"][0:3].copy()),\
                                          np.array(self.obs_dict["FT_raw"][0:3].copy())))

    def get_obs(self):
        self.obs_dict = self.controller.state_dict.copy()
        self.obs_dict['FT'] += self.controller.virtual_ext_force.copy()
        self.obs_dict['ext_force'] = self.controller.virtual_ext_force.copy()
        #state = np.concatenate((self.x_d - np.array(self.obs_dict["pose"][0:3].copy()), \
                                    #np.array(self.obs_dict["vel"][0:3].copy()), np.array(self.obs_dict["pose"][0:3].copy()) ))
        state = np.concatenate(( np.array(self.obs_dict["pose"][0:3].copy()), np.array(self.obs_dict["vel"][0:3].copy()), \
                                 self.x_d - np.array(self.obs_dict["pose"][0:3].copy()),\
                                          np.array(self.obs_dict["FT_raw"][0:3].copy())))
        return state

    def get_reward(self):
        #self.obs_dict = self.controller.state_dict
        x_reward = np.exp(-np.square(300 * (self.obs_dict["pose"][0] - self.x_d)))
        y_reward = np.exp(-np.square(300 * (self.obs_dict["pose"][1] - self.x_d)))
        reward = + 0*x_reward + 0*y_reward
        return reward

    def get_reward_basic(self, achieved_goal, desired_goal, obs, input, prev_input):
        obs_cost = (np.square(100*(desired_goal - achieved_goal)))
        act_cost =1*(np.square(input))
        if prev_input is not None:
            smooth_cost = np.sum(1 * np.square(input - prev_input))
        else:
            smooth_cost = 0
        # reward = torch.exp(-torch.sum((10*next_obs[:,0:3] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
        #print(obs_cost, act_cost)
        reward = -(100*np.abs(obs[6])*obs_cost[0] + 100*np.abs(obs[7])*obs_cost[1] + 100*np.abs(obs[8])*obs_cost[2]+\
                   (20-0*np.abs(obs[9]))*act_cost[0] + (20-0*np.abs(obs[10]))*act_cost[1] + \
                   (20-np.abs(obs[11]))*act_cost[2] + 10 * smooth_cost)
        #reward = -(1*obs_cost + 0*act_cost)
        return reward

    def her_reward(self, achieved_goal, desired_goal, input):
        r_track = -goal_distance(20 * achieved_goal, 20 * desired_goal)
        r_k = -np.mean(np.square(input), axis=-1)
        reward = r_track  # 1*r_k
        return reward

    def change_goal(self):
        if self.goal_type == "random":
            # obs_dict = self.controller.state_dict.copy()
            current_pose = self.obs_dict['pose'][0:3].copy()
            self.x_d = current_pose #+ np.random.uniform(low=-.1, high=.1, size=(3,))  # np.array([0.2,0.2,0.2])#
            #self.x_d = np.asarray(self.robot.ee_pose()[0])
            #self.x_d += np.array([0, 0, 0.0])
            #self.x_d_dot = np.zeros(6)
            #self.x_d_ddot = np.zeros(6)

    def get_goal(self):
        return self.x_d.copy(), np.diag(np.array(self.obs_dict['K']))[0:3]

    def get_action(self):
        return self.action

    def step(self, action):
        #if self.position_as_action:
        x_dot = self.obs_dict['vel'][0:3].copy()
        self.action = action.copy()
        K_target = 3000 * action
        K_0 = np.diag(self.obs_dict['K'])[0:3].copy()
        delta_K = K_target - K_0
        K = np.diag(self.controller.state_dict['K'])[0:3].copy()
        for k in range(100):
            K[0:3] = K_0 + delta_K * (1 - np.exp(-k / 20))
            self.controller.set_goal(K, self.x_d, self.goal_ori, 0 * self.x_d_dot, 0 * self.x_d_ddot)
            self.controller._send_cmd()
            if self.render_robot:
                # print("rendering")
                self.robot.render()

        self.controller.timestep += 1
        last_goal = self.x_d.copy()
        self.i += 1
        obs = self.get_obs()
        #self.change_goal()
        pose = self.obs_dict['pose'][0:3].copy()
        x_ddot = self.robot_acceleration(x_dot)

        if self.i==30:
            ball_pose = np.concatenate((self.robot.body_pose("ball1")[0], self.robot.body_pose("ball1")[1]))
            ball_pose[0] = 0.36
            ball_pose[1] = 00.085
            ball_pose[2] = 1.7 + 0.1*np.random.rand()
            self.robot.hard_set_joint_positions(ball_pose, np.arange(7, 14))
        if self.i==70:
            ball_pose = np.concatenate((self.robot.body_pose("ball2")[0], self.robot.body_pose("ball2")[1]))
            ball_pose[0] = 0.36
            ball_pose[1] = 0.085
            ball_pose[2] = 1.7 + 0.1*np.random.rand()
            self.robot.hard_set_joint_positions(ball_pose, np.arange(14, 21))
        if self.i==110:
            ball_pose = np.concatenate((self.robot.body_pose("ball3")[0], self.robot.body_pose("ball3")[1]))
            ball_pose[0] = 0.36
            ball_pose[1] = 0.085
            ball_pose[2] = 1.7 + 0.1*np.random.rand()
            self.robot.hard_set_joint_positions(ball_pose, np.arange(21, 28))
        if self.i==150:
            ball_pose = np.concatenate((self.robot.body_pose("ball4")[0], self.robot.body_pose("ball4")[1]))
            ball_pose[0] = 0.36
            ball_pose[1] = 0.085
            ball_pose[2] = 1.7 + 0.1*np.random.rand()
            self.robot.hard_set_joint_positions(ball_pose, np.arange(28, 35))
        if self.i==-100:
            ball_pose = np.concatenate((self.robot.body_pose("ball1")[0], self.robot.body_pose("ball1")[1]))
            ball_pose[2] = 0
            self.robot.hard_set_joint_positions(ball_pose, np.arange(7, 14))

        if (self.i >= self.max_num_it)  :
            done = True#(self.iteration >= self.max_num_it)
        else:
            done = False
        reward = self.get_reward_basic(last_goal, pose, obs, self.action.copy(), self.prev_action )
        info = {}
        #print(self.robot.get_ft_reading())
        #print("torque ", self.controller._cmd)
        #print("pose ", self.x_d[:, self.i-1] - self.robot.ee_pose()[0])
        #print("smoothed FT reading: ", self.obs_dict['FT_raw'])
        #print(self.i)
        #print(self.robot.body_pose("object1")[0])
        #print(self.robot._sim.data.qpos)
        self.prev_action = self.action.copy()
        return obs, reward, done, info

    def reset(self):
        print("resetting envs")
        self.i = 0
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.hard_set_joint_positions(self.object1_init_pose, np.arange(7,14))
        self.robot.hard_set_joint_positions(self.object2_init_pose, np.arange(14, 21))
        self.robot.hard_set_joint_positions(self.object3_init_pose, np.arange(21, 28))
        self.robot.hard_set_joint_positions(self.object4_init_pose, np.arange(28, 35))
        self.robot.sim_step()
        self.controller.reset()
        self.obs_dict = self.controller.state_dict.copy()
        self.change_goal()
        return self.get_obs()

    def render(self):
        self.robot.render()

def make_panda_tray_env(env_cfg):
    #num_it = env_cfg.overrides.get("trial_length", cfg.MAX_NUM_IT)
    controller = env_cfg.overrides.get("controller", "VIC")
    #control_rate = env_cfg.overrides.get("control_rate", cfg.PUBLISH_RATE)
    render = env_cfg.get("render", True)
    env = PandaTrayEnv(controller=controller,  render=render)
    return env

def plot(traj, goal, k, f_ext, f, acc ):
    #k =k.astype(int)
    #print(k)
    for i in range (12):
        plt.subplot(4,3,i+1)
        if i in (0,1,2):
            plt.plot(100*traj[:, i])
            plt.plot(100*goal[:, i])
        elif i  in (3,4,5):
            plt.plot(k[:, i-3])
        elif i in (6,7,8):
            plt.plot(f_ext[:, i-6])
        elif i in (9,10,11):
            plt.plot(f[:, i - 9])
        #else:
            #plt.plot(acc[:, i - 12])
    plt.show()

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR, SAC
from stable_baselines.common.cmd_util import make_vec_env
if __name__ == "__main__":
    train = True
    restart = True
    VIC_env = PandaTrayEnv(controller = "VIC") #
    #VIC_env.set_render(True)


    if train:
        env = make_vec_env(lambda: VIC_env, n_envs=1)
        if restart:
            model = SAC.load('Policies/panda_tray/panda_tray')
            model.set_env(env)
        else:
            model = SAC('MlpPolicy', env, verbose=1)#.learn(100000)
        model.learn(total_timesteps=100000, log_interval=10)
        model.save('Policies/panda_tray/panda_tray_100pos_20act')
    else:
        VIC_env.set_render(False)
        env = VIC_env#make_vec_env(lambda: VIC_env, n_envs=1)
        obs = env.reset()
        model = SAC.load('Policies/panda_tray/panda_tray')
        n_steps = 200
        episodes = 100
        current_trial = 0
        output_dir = "/home/akhil/PhD/Publications/COnferences/CoRL2022/Figures/SAC_VIL/tray/Data"
        filename = "tray"
        save = True
        plot_results = True
        for i in range(episodes):
            obs = env.reset()
            done = False
            goal = []
            stiffness = []
            observations = []
            acceleration = []
            cartesian_force = []
            ext_force = []
            data = {}

            extra_obs = env.get_extra_obs()
            g, K = env.get_goal()
            observations.append(extra_obs['pose'][0:3])
            goal.append(g)
            stiffness.append(np.diag(extra_obs['K'])[0:3])
            # print(np.diag(extra_obs['K'])[0:3])
            # acceleration.append(extra_obs['acceleration'])
            cartesian_force.append(extra_obs['cartesian_force'][0:3])
            ext_force.append(0 * extra_obs['FT'][0:3])
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                #print("Step {}".format(step + 1))
                #print("Action: ", action)
                obs, reward, done, info = env.step(action)
                extra_obs = env.get_extra_obs()
                #print('obs=', obs, 'reward=', reward, 'done=', done)
                #env.render()
                if done:
                    data['o'] = np.array(observations)
                    data['g'] = np.array(goal)
                    data['k'] = np.array(stiffness)
                    data['f_ext'] = np.array(ext_force)
                    data['f_cartesian'] = np.array(cartesian_force)
                    data['acc'] = np.array(acceleration)
                    filename_trial = filename + '_' + str(current_trial)
                    if save:
                        np.save(os.path.join(output_dir, filename_trial), data)
                    if plot_results:
                        plot(np.array(observations), np.array(goal), np.array(stiffness),\
                             np.array(ext_force), np.array(cartesian_force), np.array(acceleration))
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                else:
                    observations.append(extra_obs['pose'][0:3])
                    goal.append(g)
                    stiffness.append(np.diag(extra_obs['K'])[0:3])
                    # print(np.diag(extra_obs['K'])[0:3])
                    acceleration.append(extra_obs['acceleration'])
                    cartesian_force.append(extra_obs['cartesian_force'][0:3])
                    ext_force.append(extra_obs['FT'][0:3])
            current_trial += 1
    #sjs
    #gym.make("gym_robotic_ultrasound:ultrasound-v0")


    '''
    curr_ee, curr_ori = VIC_env.robot.ee_pose()
    print(VIC_env.robot.ee_pose())
    # --------------------------------------

    VIC_env.controller.set_active(True)
    now_r = time.time()
    i = 0
    count = 0
    Data = []
    VIC_env.reset()
    VIC_env.set_render(True)

    for i in range(1):
        #print(VIC_env.robot.body_pose("object1"))
        print(i)
    robot_pos, robot_ori = VIC_env.robot.ee_pose()
    while True:
        # get current robot end-effector pose
        timestep = VIC_env.controller.timestep / 1
        #print("helloooo",i, timestep)

        #render_frame(VIC_env.robot.viewer, robot_pos, robot_ori)
        #render_frame(VIC_env.robot.viewer, VIC_env.x_d[:, i], VIC_env.goal_ori, alpha=0.2)
        if True:
            elapsed_r = time.time() - now_r
            # render controller target and current ee pose using frames
            action = np.array([1, 1, 1])
            #action["xd"] = robot_pos# + np.array([0.05,0.05,0])
            #action = np.array([1,0.5,2])# 10*(1-2*np.random.random())#0#.000001#np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s,r,done,_ = VIC_env.step(action)
            print("reward: ", r)
            print(VIC_env.robot.get_ft_reading()[0])
            #print(s)
            if done:
                VIC_env.reset()
            i += 1
            if i == 4999:
                break
                #break
            print(VIC_env.robot.ee_pose())
        #VIC_env.robot.render()  # render the visualisation

    plt.plot(VIC_env.controller.F_history[:,2])
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()'''