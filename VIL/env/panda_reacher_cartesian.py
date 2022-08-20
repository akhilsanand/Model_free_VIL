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
from mujoco_panda.controllers.torque_based_controllers.VIC_env_configs import VIC_reacher_cartesian_config as cfg
import time
import random
import quaternion
import matplotlib.pyplot as plt


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class PandaReacherCartesian(gym.Env):#(modified_gym.GoalEnv):#

    def __init__(self, position_as_action = False, controller = "VIC", reward_type= "dense", pert_type='none' \
                 , n_actions= 3,  log_dir = None, render=False, goal_type = 'random'):

        MODEL_PATH = os.environ['MJ_PANDA_PATH'] + '/mujoco_panda/models/'
        #self.robot = PandaArm(model_path=MODEL_PATH + 'panda_block_table.xml',
                             #render=True, compensate_gravity=False, smooth_ft_sensor=True)
        self.robot = PandaArm(model_path=MODEL_PATH + 'franka_panda_no_gripper.xml',
                             render=render, compensate_gravity=True, smooth_ft_sensor=True)

        if mujoco_py.functions.mj_isPyramidal(self.robot.model):
            print("Type of friction cone is pyramidal")
        else:
            print("Type of friction cone is eliptical")
        # self.init_jpos = np.array([-0.0242187,-0.26637015,-0.23036408,-1.99276073,-0.05414588,1.72812007, 0.52766157])
        #self.init_jpos = np.array([0., -0.7, 0, -2.356, 0, 1.656, 0.785])
        self.init_jpos = np.array([-2.03e-03, -8.42e-01, 1.09e-03, -2.26e+00, 5.80e-04, 1.41e+00, 8.24e-01])
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
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
        self.x_d = np.asarray(self.robot.ee_pose()[0].copy()) + np.array([0,0,0])
        self.x_d_dot = np.zeros(6)
        self.x_d_ddot = np.zeros(6)
        self.action = None
        #data = np.load('/home/akhil/PhD/Publications/COnferences/Humanoids2022/Figures/Experiments/Door/Data/Data.npy', encoding='latin1', allow_pickle=True)[()]
        #self.f_data = data['f_ext']
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
        #self.observation_space = spaces.Box(
            #low=np.array([cfg.LOWER_Fz, cfg.LOWER_Z_ERROR, cfg.LOWER_Kz, cfg.LOWER_Fz]), \
            #high=np.array([cfg.UPPER_Fz, cfg.UPPER_Z_ERROR, cfg.UPPER_Kz, cfg.UPPER_Fz]))

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
        return np.concatenate((self.x_d - np.array(self.obs_dict["pose"][0:3].copy()),\
                                          np.array(self.obs_dict["FT_raw"][0:3].copy())))#self.obs_dict["j_pose"].copy()#(self.controller.virtual_ext_force.copy()[0:3])

    def get_obs(self):
        self.obs_dict = self.controller.state_dict.copy()
        self.obs_dict['FT'] += self.controller.virtual_ext_force.copy()
        self.obs_dict['ext_force'] = self.controller.virtual_ext_force.copy()
        state = np.concatenate((np.array(self.obs_dict["pose"][0:3].copy()), np.array(self.obs_dict["vel"][0:3].copy()),
                                self.x_d - np.array(self.obs_dict["pose"][0:3].copy()), \
                                np.array(self.obs_dict["FT_raw"][0:3].copy())))
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

    def get_reward_basic(self, achieved_goal, desired_goal, obs, input):
        obs_cost = (np.square(100*(desired_goal - achieved_goal)))
        act_cost =1*(np.square(input))
        # reward = torch.exp(-torch.sum((10*next_obs[:,0:3] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
        #print(obs_cost, act_cost)
        reward = -(10*np.abs(obs[6])*obs_cost[0] + 30*np.abs(obs[7])*obs_cost[1] + 10*np.abs(obs[8])*obs_cost[2]+\
                   (11-np.abs(obs[9]))*act_cost[0] + (11-np.abs(obs[10]))*act_cost[1] + \
                   (11-np.abs(obs[11]))*act_cost[2])
        #reward = -(1*obs_cost + 0*act_cost)
        return reward

    #for HER
    def her_reward(self, achieved_goal, desired_goal, input):
        #r_track = np.exp(-goal_distance(30*achieved_goal, 30*desired_goal)) -1
        r_track = -goal_distance(20 * achieved_goal, 20 * desired_goal)
        r_k = -np.mean(np.square(input), axis = -1)
        #print("test: ", input, input+1)
        reward = r_track#1*r_k
        return reward

    def change_goal(self):
        if self.goal_type == "random":
            #obs_dict = self.controller.state_dict.copy()
            current_pose = self.obs_dict['pose'][0:3].copy()
            self.x_d     = current_pose + np.array([0,0,0])# 0*np.random.uniform(low=-.1, high=.1, size=(3,)) #np.array([0.2,0.2,0.2])#
        #self.x_d = current_pose

    def get_goal(self):
        return self.x_d.copy(), np.diag(np.array(self.obs_dict['K']))[0:3]

    def get_action(self):
        return self.action

    def get_sin_force(self, offset, duration):
        f = 10* np.sin(2*np.pi*(self.i-offset)/duration) + 5*(2*random.random() - 1)
        #fy = 10 * np.sin(2 * np.pi * (self.i - offset) / duration) + 5*(2 * random.random() - 1)
        #fz = 10 * np.sin(2 * np.pi * (self.i - offset) / duration) + 5*(2 * random.random() - 1)
        return f

    def step(self, action):
        #print(action)
        #print(self.i)
        #print(self.goal_ori )
        x_dot = self.obs_dict['vel'][0:3].copy()
        '''
        if (self.i) % 1 == 0:
            if np.random.rand() >= 0.1:
                virtual_ext_force = 1*np.concatenate((np.random.uniform(low=-10, high=10, size=(3,)), np.zeros(3)))
            else:
                virtual_ext_force = np.zeros(6)

        if self.i<20:
            virtual_ext_force = np.zeros(6)
        elif self.i<30:
            virtual_ext_force = np.array([10, 0, -10, 0, 0, 0])
        elif self.i < 40:
            virtual_ext_force = np.array([5, -5, -5, 0, 0, 0])
        elif self.i < 50:
            virtual_ext_force = np.array([0, -10, 0, 0, 0, 0])
        elif self.i < 60:
            virtual_ext_force = np.array([-5, 0, 5, 0, 0, 0])
        elif self.i < 70:
            virtual_ext_force = np.array([-10, 5, 10, 0, 0, 0])
        elif self.i < 80:
            virtual_ext_force = np.array([0, 10, 0, 0, 0, 0])
        else:
            virtual_ext_force = np.zeros(6)
        '''

        if self.i < 10:
            virtual_ext_force = np.zeros(6)
        elif self.i <30:
            virtual_ext_force = np.array([self.get_sin_force(10,50), 0, -self.get_sin_force(10,50),0,0,0])
        elif self.i < 60:
            virtual_ext_force = np.array([self.get_sin_force(10, 50), self.get_sin_force(30, 50), -self.get_sin_force(10, 50), 0, 0, 0])
        elif self.i < 80:
            virtual_ext_force = np.array([0, self.get_sin_force(30, 50), 0, 0, 0, 0])
        else:
            virtual_ext_force = np.zeros(6)

        #virtual_ext_force = np.array([0, 0, 0, 0, 0, 0])
        #virtual_ext_force[0:3] = self.f_data[9*self.i,:]
        self.robot.apply_external_force_on_ee(virtual_ext_force)
        #action = np.array([1,1,1])
        self.action = action.copy()
        K_target = 3000 * action
        K_0 = np.diag(self.obs_dict['K'])[0:3].copy()
        delta_K = K_target - K_0
        K = np.diag(self.controller.state_dict['K'])[0:3].copy()
        for k in range(100):
            K = K_0 + delta_K * (1 - np.exp(-k / 20))
            self.controller.set_goal(K, self.x_d, self.goal_ori, 0*self.x_d_dot, 0*self.x_d_ddot)
            self.controller._send_cmd()
            if self.render_robot:
                #print("rendering")
                self.robot.render()

        self.controller.timestep += 1
        last_goal = self.x_d.copy()
        self.i += 1
        #print(self.i)
        obs = self.get_obs()
        #self.change_goal()
        pose = self.obs_dict['pose'][0:3].copy()
        x_ddot = self.robot_acceleration(x_dot)
        if (self.i >= 100):
            done = True#(self.iteration >= self.max_num_it)
        else:
            done = False
        cartesian_force = self.obs_dict['cartesian_force'][0:3].copy()
        reward = self.get_reward_basic(last_goal, pose, obs, action )
        info = {}
        return obs, reward, done, info

    def reset(self):
        print("resetting envs")
        self.i = 0
        self.robot.hard_set_joint_positions(self.init_jpos)
        self.robot.sim_step()
        self.controller.reset()
        self.obs_dict = self.controller.state_dict.copy()
        self.change_goal()

        return self.get_obs()

    def render(self, mode=None, **kwargs):
        self.robot.render()

def make_panda_reacher_cartesian_env(env_cfg):
    #num_it = env_cfg.overrides.get("trial_length", cfg.MAX_NUM_IT)
    controller = env_cfg.overrides.get("controller", "VIC")
    #control_rate = env_cfg.overrides.get("control_rate", cfg.PUBLISH_RATE)

    render = env_cfg.get("render", True)

    env = PandaReacherCartesian(controller=controller,  render=render)

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

from stable_baselines import DQN, PPO2, A2C, ACKTR, SAC
from stable_baselines.common.cmd_util import make_vec_env

if __name__ == "__main__":
    train = False
    restart = True
    VIC_env = PandaReacherCartesian(controller = "VIC") #
    if train:
        env = make_vec_env(lambda: VIC_env, n_envs=1)
        if restart:
            model = SAC.load('Policies/panda_hold/panda_hold') # panda_cartesian_reacher
            model.set_env(env)
        else:
            model = SAC('MlpPolicy', env, verbose=1)#.learn(100000)
        model.learn(total_timesteps=2000, log_interval=10)
        model.save('Policies/panda_hold/panda_hold')
    else:
        VIC_env.set_render(False)
        env = VIC_env#make_vec_env(lambda: VIC_env, n_envs=1)
        obs = env.reset()
        model = SAC.load('Policies/panda_hold/panda_hold')
        n_steps = 200
        episodes = 100
        current_trial = 0
        output_dir = "/home/akhil/PhD/Publications/COnferences/CoRL2022/Figures/SAC_VIL/hold_pose/Data"
        filename = "hold"
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
                # print("Step {}".format(step + 1))
                # print("Action: ", action)
                obs, reward, done, info = env.step(action)
                extra_obs = env.get_extra_obs()
                # print('obs=', obs, 'reward=', reward, 'done=', done)
                # env.render()
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
                        plot(np.array(observations), np.array(goal), np.array(stiffness), \
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
    '''
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
    VIC_env.set_render(True)
    #VIC_env.robot.render = True
    #VIC_env.robot.render()
    print(i)
    while True:
        # get current robot end-effector pose
        timestep = VIC_env.controller.timestep / 1
        #print("helloooo",i, timestep)
        robot_pos, robot_ori = VIC_env.robot.ee_pose()
        while (True):
            elapsed_r = time.time() - now_r
            # render controller target and current ee pose using frames
            action = np.array([1,1,1])# 10*(1-2*np.random.random())#0#.000001#np.array([0.0001, 0.00001])  # np.random.uniform(1.e-6, 0.01, 2)
            s,r,done,_ = VIC_env.step(action)
            print("reward: ", r)
            #if i < 10:
             #   VIC_env.robot.apply_external_force_on_ee(1*np.array([1,10,-10,0,0,0]))
            #else:
             #   VIC_env.robot.apply_external_force_on_ee(1 * np.array([0, 0,0, 0, 0, 0]))
            #print(s)
            if done:
                VIC_env.reset()
                #VIC_env.set_render(True)
                #VIC_env.robot.render()
            i += 1
            if i == 4999:
                break
        #VIC_env.robot.render()  # render the visualisation

    plt.plot(VIC_env.controller.F_history[:,2])
    plt.show()
    # input("Trajectory complete. Hit Enter to deactivate controller")
    VIC_env.controller.set_active(False)
    VIC_env.controller.stop_controller_cleanly()'''