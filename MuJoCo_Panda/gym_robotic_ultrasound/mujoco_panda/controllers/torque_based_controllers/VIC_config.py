import gym
#from gym import ...

#! /usr/bin/envs python
import numpy as np
import gym
from gym import spaces
#from gym_panda.envs import VIC_func as func

SIM_STATUS = True
ADD_NOISE = True
NOISE_FRACTION = 0.015 #standard deviation of the noise is now 1.5 % of the force-value

Fd = 10
ROBOT_CONTROL_RATE = 10
PUBLISH_RATE = 10
duration = 5
z_max = 0.584#5.91776e-01
z_min = 0.582


#ACTION SPACE  RANDOM VALUES
GAMMA_B_LOWER = 1.e-7#0.0001#10**(-3)
GAMMA_B_UPPER = 0.01#10**(-1)

GAMMA_K_LOWER = 1.e-7# 0.01#0.000001#10**(-4)
GAMMA_K_UPPER = 0.01#0.0001#10**(-2)

KP_POS_LOWER = 500
KP_POS_UPPER =  1000

Kz_LOWER = -1
Kz_UPPER = 1
Bz_LOWER = -1
Bz_UPPER = 1


#initialization

GAMMA_B_INIT = 0.001*10 # 10**(-2) #never applied
GAMMA_K_INIT = 0.0005/10#10**(-2) #never applied



# parameters of stiffness and damping matrices
Kp =  2000#10000#1250
#Kpz = 1000#300#20#35#50 #initial value (adaptive)
Ko = 0#10000#25000#5000#1500#900

Bp = 2*np.sqrt(Kp)#700/4
#Bpz = np.sqrt(Kpz)#10 # #initial value (adaptive)
Bo = 2*np.sqrt(Ko)#10#0# 3750 #10#100#10

# MASS, DAMPING AND STIFFNESS MATRICES (ONLY M IS COMPLETELY CONSTANT)
M = np.identity(6)*1

B = np.array([[Bp, 0, 0, 0, 0, 0],
                [0, Bp, 0, 0, 0, 0],
                [0, 0, Bp, 0, 0, 0],
                [0, 0, 0, Bo, 0, 0],
                [0, 0, 0, 0, Bo, 0],
                [0, 0, 0, 0, 0, Bo]])
K = np.array([[Kp, 0, 0, 0, 0, 0],
                [0, Kp, 0, 0, 0, 0],
                [0, 0, Kp, 0, 0, 0],
                [0, 0, 0, Ko, 0, 0],
                [0, 0, 0, 0, Ko, 0],
                [0, 0, 0, 0, 0, Ko]])
'''
B = np.array([[175, 0, 0, 0, 0, 0],
                [0, 156, 0, 0, 0, 0],
                [0, 0, Bpz, 0, 0, 0],
                [0, 0, 0, 400, 0, 0],
                [0, 0, 0, 0, 400, 0],
                [0, 0, 0, 0, 0, 400]])
K = np.array([[1000, 0, 0, 0, 0, 0],
                [0, 1000, 0, 0, 0, 0],
                [0, 0, Kpz, 0, 0, 0],
                [0, 0, 0, 1000, 0, 0],
                [0, 0, 0, 0, 1000, 0],
                [0, 0, 0, 0, 0, 1000]])'''


K_v = np.identity(6)
P = np.identity(6)

B_hat_lower = 0
B_hat_upper = 30000#300
B_hat_limits = [B_hat_lower,B_hat_upper]

K_hat_lower = 0#10
K_hat_upper = 2500000#1000
K_hat_limits = [K_hat_lower,K_hat_upper]

#list_of_limits = [GAMMA_B_LOWER, GAMMA_B_UPPER, GAMMA_K_LOWER,GAMMA_K_UPPER, KP_POS_LOWER, KP_POS_UPPER,B_hat_lower,B_hat_upper,K_hat_lower,K_hat_upper ]




T = 0.001*(1000/PUBLISH_RATE) # The control loop's time step
MAX_NUM_IT = int(duration*PUBLISH_RATE)
#ALTERNATIVE_START = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, 'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}
'''
ALTERNATIVE_START = {'panda_joint1': -0.01780731604828034, 'panda_joint2': -0.7601326257410115, 'panda_joint3': 0.019760083535855344, \
                     'panda_joint4': -2.342100576747406, 'panda_joint5': 0.029903952654724897, 'panda_joint6': 1.541202406072693, \
                     'panda_joint7': 0.7535284559526767}'''
cartboard = {'panda_joint1': 1.5100039307153879, 'panda_joint2': 0.6066719992230666, 'panda_joint3': 0.024070900507747097, 'panda_joint4': -2.332000750114692, \
            'panda_joint5': -0.037555063873529436, 'panda_joint6': 2.9529732850154575, 'panda_joint7': 0.7686490028450895}

ALTERNATIVE_START =  {'panda_joint1': -0.022383421807846915, 'panda_joint2': -0.7741431570618871, 'panda_joint3': 0.007015706042152736, 'panda_joint4': -2.357305725939209, \
                     'panda_joint5': 0.020046381777886424, 'panda_joint6': 1.5541264226404703, 'panda_joint7': 0.7508477102109792}

RED_START = {'panda_joint1':-0.020886360413928884, 'panda_joint2':-0.6041856795321063, 'panda_joint3': 0.022884284694488777, 'panda_joint4': -2.241203921591765, 'panda_joint5': 0.029363915766836612, 'panda_joint6': 1.5962793070668644, 'panda_joint7': 0.7532362527093444}
#OBSERVATION SPACE


UP = {'panda_joint1':-0.011832780553287847, 'panda_joint2':-0.6745771298364058, 'panda_joint3': 0.04155051269907606, 'panda_joint4': -2.0013764007695816, 'panda_joint5': 0.05021809784675746, 'panda_joint6': 1.3726852401919203, 'panda_joint7': 0.7624296573975551}

LOWER_F = -20
UPPER_F = 30

LOWER_Fx = -100
LOWER_Fy = -100
LOWER_Fz = -100
UPPER_Fx = 100
UPPER_Fy = 100
UPPER_Fz = 100

LOWER_Kx = -100
LOWER_Ky = -100
LOWER_Kz = K_hat_lower
UPPER_Kx = 100
UPPER_Ky = 100
UPPER_Kz = K_hat_upper

LOWER_Tx = -10
LOWER_Ty = -10
LOWER_Tz = -10
UPPER_Tx = 10
UPPER_Ty = 10
UPPER_Tz = 10

LOWER_Vx = -2
LOWER_Vy = -2
LOWER_Vz = -2
UPPER_Vx = 2
UPPER_Vy = 2
UPPER_Vz = 2

LOWER_Ax = -5
LOWER_Ay = -5
LOWER_Az = -5
UPPER_Ax = 5
UPPER_Ay = 5
UPPER_Az = 5

DELTA_Z_LOWER = -0.1
DELTA_Z_UPPER = 0.1

DELTA_K_LOWER = -1
DELTA_K_UPPER = 1

LOWER_VEL = -10
UPPER_VEL = 10

LOWER_X_ERROR = -0.1
UPPER_X_ERROR = 0.1

LOWER_Y_ERROR = -0.1
UPPER_Y_ERROR = 0.1

LOWER_Z_ERROR = -0.1
UPPER_Z_ERROR = 0.1


LOWER_FORCE_DOT = -1000
UPPER_FORCE_DOT = 1000

LOWER_FORCE_OVERSHOOT = 0
UPPER_FORCE_OVERSHOOT = 50


