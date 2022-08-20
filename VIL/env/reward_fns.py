# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6 ** 2))
    act_cost = -0.01 * torch.sum(act ** 2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)
    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]
    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)

def ultrasound(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    Fd = torch.tensor([100]).to(next_obs.device)
    F_reward = (next_obs[:, 2] - Fd) ** 2
    #F_reward = torch.exp(-torch.sum(3*F_reward ** 2, dim=1))
    pose_reward = torch.sum((100 * next_obs[:, 3:5] ** 2), dim=1)
    act_cost = 0 * (act ** 2).sum(axis=1)
    reward = -(0*F_reward + 1*pose_reward + 0*act_cost)

    #print(reward.view(-1, 1))
    return (reward).view(-1, 1)


def HFMC1(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    Fd= torch.tensor([3]).to(next_obs.device)
    f_z = next_obs[:, 0]
    print("Fd = ", Fd, "fz = ", f_z)
    delta_f = (Fd - f_z).abs().sum(axis=1)
    sq_error = torch.square(delta_f)
    return -(sq_error).view(-1, 1)

def panda_traj_tracking(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    #for 5 states
    #assert len(next_obs.shape) == len(act.shape) == 2
    #reward_ctrl = -0.1 * act.square().sum(dim=1)
    obs_cost = torch.sum((100 * next_obs[:, 0:3] ** 2), dim=1)
    #obs_cost = 10*next_obs[:,0:3].abs().sum(axis=1)
    act_cost = 0 * (act ** 2).sum(axis=1)
    #reward = torch.exp(-torch.sum((10*next_obs[:,0:3] ** 2), dim=1))#np.exp(-np.square(next_obs[3]))
    reward  = -(1*obs_cost + 1*act_cost)
    return reward.view(-1, 1)

def panda_reacher_cartesian(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:

    #obs_cost = 1 * torch.sum(torch.square(100*(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])), dim= 1)
    obs_cost = torch.square(100 * (act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3]))
    oc_x, oc_y, oc_z = obs_cost[:,0], obs_cost[:,1], obs_cost[:,2]
    act_cost = 1 * (act[:, 0:3] ** 2)
    ac_x, ac_y, ac_z =  act_cost[:,0], act_cost[:,1], act_cost[:,2]

    acc_cost = torch.sum((100*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
    else:
        smooth_cost = 0
    #act_cost = 1 * (act[:,0:3] ** 2).sum(axis=1)
    reward = -(10*torch.abs(act[0,3])*oc_x + 30*torch.abs(act[0,4]) *oc_y + 10*torch.abs(act[0,5])*oc_z + \
              (11-torch.abs(act[0,6]))*ac_x + (11-torch.abs(act[0,7]))*ac_y + (11-torch.abs(act[0,8]))*ac_z + \
               0 * acc_cost + 0 * smooth_cost)
    '''if (torch.sum(torch.abs(act[0,6:9])) > 0.1 or torch.sum(torch.abs(act[0,3:6])) > 0.005 ):
        reward = -(10*oc_x + 10*oc_y + 10*oc_z + 1 * ac_x + 1 * ac_y + 1 * ac_z + 0 * acc_cost + 0 * smooth_cost)
    else:
        reward = -(1*oc_x + 1*oc_y + 1*oc_z + 10*ac_x + 10*ac_y + 10*ac_z + 0 * acc_cost + 0 * smooth_cost)'''
    #reward = -(10 * oc_x + 10 * oc_y + 10 * oc_z + 1 * ac_x + 1 * ac_y + 1 * ac_z + 0 * acc_cost + 0 * smooth_cost)
    #print(act[0,6])
    #print("reward:", reward)
    return reward.view(-1, 1)

def panda_tray(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    obs_cost = torch.square(100 * (act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3]))
    #r_scale = torch.abs(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])
    oc_x, oc_y, oc_z = obs_cost[:, 0], obs_cost[:, 1], obs_cost[:, 2]
    act_cost = 1 * (act[:, 0:3] ** 2)
    ac_x, ac_y, ac_z = act_cost[:, 0], act_cost[:, 1], act_cost[:, 2]
    acc_cost = torch.sum((100*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
    else:
        smooth_cost = 0
    reward  = -(1*torch.sum(obs_cost,dim=1) + 1*act_cost.sum(axis=1) + 0 * acc_cost + 0*smooth_cost)
    #print("prior ", obs_cost)
    '''if (act[0,8] < -13):
        obs_cost[:,2] = 100*obs_cost[:,2]
        act_cost[:,0:2] = 10*act_cost[:,0:2]
        #print(obs_cost)
        reward = -(1*torch.sum(obs_cost,dim=1) + 1*act_cost.sum(axis=1) + 0 * acc_cost + 1*smooth_cost)
        #print("reward:", reward)''' #r_scale[:,2]
    reward = -(100*torch.abs(act[0,3])*oc_x + 100*torch.abs(act[0,4])*oc_y + 100*torch.abs(act[0,5])*oc_z + \
               + (10-0*torch.abs(act[0,6]))*ac_x +(10-0*torch.abs(act[0,7]))*ac_y+(10-torch.abs(act[0,8]))*ac_z+
               0 * acc_cost + 10 * smooth_cost)
    return reward.view(-1, 1)

def panda_pusher(act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    obs_cost = torch.square(100 * (act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3]))
    #r_scale = torch.abs(act[:, 3:6] + pre_obs[:, 0:3] - next_obs[:, 0:3])
    oc_x, oc_y, oc_z = obs_cost[:, 0], obs_cost[:, 1], obs_cost[:, 2]
    act_cost = 1 * (act[:, 0:3] ** 2)
    ac_x, ac_y, ac_z = act_cost[:, 0], act_cost[:, 1], act_cost[:, 2]
    acc_cost = torch.sum((100*(next_obs[:, 3:6]-pre_obs[:,3:6])**2), dim= 1)
    if pre_act is not None:
        smooth_cost = torch.sum((1*(act[:, 0:3]-pre_act[:,0:3])**2), dim= 1)
    else:
        smooth_cost = 0

    #reward  = -(1*obs_cost + 1*act_cost + 0 * acc_cost + 0*smooth_cost)
    reward = -(1*oc_x + 1*oc_y + 00*oc_z + \
               + 1*(10-torch.abs(act[0,6]))*ac_x +1*(10-torch.abs(act[0,7]))*ac_y+0*ac_z+
               10 * acc_cost + 1 * smooth_cost)
    #print("reward:", reward)
    return reward.view(-1, 1)

def panda_touch (act: torch.Tensor, next_obs: torch.Tensor, pre_obs: torch.Tensor,  pre_act: torch.Tensor ) -> torch.Tensor:
    delta_pose =  act[:, 5] #+ pre_obs[:, 2] - next_obs[:, 2]  #  as the next observation are not ideal to use as tehre is no movement of the robotin contact direction
    #the deltapose is based on the feedback from env, the current velocity read from env is used as the next prections are not suitable in contact direction
    cartesian_force =  -1*(3000*act[:, 2] * delta_pose - 2* torch.sqrt(3000*act[:, 2]) * pre_obs[:, 5]) #
    force_reward = torch.square(cartesian_force - 50) #
    obs_cost = 1 * torch.sum(torch.square(1000*(act[:, 3:5] + pre_obs[:, 0:2] - next_obs[:, 0:2])), dim= 1)
    act_cost = 1 * (act[:, 0:3] ** 2).sum(axis=1)
    reward  = -(0*obs_cost + 1*force_reward + 0.1*act_cost)
    #print("cartesian force_predicted : ", (cartesian_force[torch.argmin(force_reward)]))
    #print("force_reward : ", torch.min(force_reward))
    #print(delta_pose)
    return reward.view(-1, 1)
