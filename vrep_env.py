import numpy as np
import gym
from gym import spaces
from api import vrep
from time import sleep
import csv
import math, time

class VrepEnv:
    #초기 값들을 설정합니다.
    def __init__(self, enables_acturator_dynamics = False):

        self.observation_space = spaces.Box(low=-1, high=1, shape=(38,))

        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))


        self.vrep_client_id=vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP
        
        self.current_goal = 0.

        self.path_list = []

        while True:
            print('Connected to remote API server')
            if self.vrep_client_id != -1:
                break
            else:
                sleep(0.2)
        
        #handles
        self.left_motor_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_motorLeft', operationMode=vrep.simx_opmode_blocking)[1]
        self.right_motor_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_motorRight', operationMode=vrep.simx_opmode_blocking)[1]
        self.steeringLeft_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_steeringLeft', operationMode=vrep.simx_opmode_blocking)[1]
        self.steeringRight_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_steeringRight', operationMode=vrep.simx_opmode_blocking)[1]
        self.car_handle = vrep.simxGetObjectHandle(clientID=self.vrep_client_id,objectName='TEAM12_VehicleFrontPose', operationMode=vrep.simx_opmode_blocking)[1]
        self.goal_number = 0
        self.road_wideth = 10
        self.wheel_speed = 20

    # step function 입니다. 
    def step(self, action):
        vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
        self.setMotorSpeeds(action)
        self.prev_car_position = self.current_car_position
        self.current_car_position = self.Getcarposition()[:2]
        self.goal_distance_calculate()
        self.getLidarData()
        self.angleCheck()
        self.getReward()
        self.goal_distance_calculate()
        obs = self.lidar_data[:]
        obs.append(self.angle)
        obs.append(action[0])

        #print ('obs:', obs)
        return obs, self.reward , self.done

    # 훈련이 멈추고 다시 시작할 때 이곳에서 값들이 reset 됩니다.
    def reset(self):
 
        vrep.simxStopSimulation(clientID=self.vrep_client_id,operationMode=vrep.simx_opmode_blocking)
        self.streaming()
        time.sleep(1)
        vrep.simxStartSimulation(clientID=self.vrep_client_id,operationMode=vrep.simx_opmode_blocking)

        vrep.simxSynchronous(self.vrep_client_id,True)
        vrep.simxSynchronousTrigger(self.vrep_client_id)
        time.sleep(1)
        self.goal_number = 0
        self.delta_ack = 0
        self.getLidarData()
        self.path_list = self.readCSV()
        self.prev_car_position = self.Getcarposition()[:2]
        self.current_car_position = self.Getcarposition()[:2]
        self.angleCheck()
        self.goal_distance_calculate()
        self.done = False
        obs = self.lidar_data[:]
        obs.append(0)
        obs.append(0)

        return obs

    # 이곳에서 리워드를 설정합니다. 리워드는 크게 라이다와 관련된 부분과 차와 목표지점이 이루는 각도로 이루어져 있습니다.
    def getReward(self):
        self.finishCheck()
        
        reward_for_lidar = 0
        try:
            for i in range(len(self.lidar_data)):
                reward_for_lidar += -1/(1+math.exp((self.lidar_data[i] - 4)))
        except:
            reward_for_lidar = 0
        
        reward_for_angle = -abs(self.angle/(math.pi/2))
        reward_for_lidar = reward_for_lidar/36
        reward_for_angle = reward_for_angle*2.9
        reward_for_lidar = reward_for_lidar*6
        #print('각도 리워드:', reward_for_angle)
        #print('라이다 리워드:', reward_for_lidar)
        self.reward =  reward_for_angle + reward_for_lidar + self.reset_reward
        
        #print('각도:', self.angle)
        #print('앞 라이다', self.front_lidar)
        #print(self.goal_number)
        #print('reward:', self.reward)
        self.reward = self.reward

    # 이 곳에서 라이다 데이터를 가져옵니다. 
    def getLidarData(self):
        _, lidar_data_bin = vrep.simxGetStringSignal(clientID=self.vrep_client_id, signalName='Team12_measurement', operationMode=vrep.simx_opmode_blocking)  
        lidar_data = np.array(vrep.simxUnpackFloats(lidar_data_bin), dtype=float)
        lidar_data = lidar_data.tolist()


        self.lidar_data = []
        if len(lidar_data) == 0:
            self.lidar_data = [self.road_wideth]*36
        else:
            try:
                for i in range(len(lidar_data)):
                    self.lidar_data.append(lidar_data[i])
            except:
                self.lidar_data = [self.road_wideth]*36
        front_lidar = self.lidar_data[17:20]
        side_lidar = self.lidar_data[5:17] + self.lidar_data[20:32]
        try:
            self.front_lidar = min(front_lidar)
            self.side_lidar = min(side_lidar)
        except:
            self.front_lidar = self.road_wideth
            self.side_lidar = self.road_wideth
        
        #print('side:', self.side_lidar)
        #print('front:', self.front_lidar)


    #이 곳에서 모터를 설정합니다. 속도는 자동차 앞의 장애물 거리에 따라서 조정하고, 각도는 ddpg를 이용해서 훈련합니다.
    def setMotorSpeeds(self,action):

        self.car_l = 2.2
        self.car_d = 0.72  


        if self.front_lidar < 2.7:
            wheel_speed = -0.5*self.wheel_speed
        elif 2.7< self.front_lidar < 4.5:
            wheel_speed =self.wheel_speed*0.5
        else:
            wheel_speed = self.wheel_speed
        #print('바퀴 속도:', wheel_speed)
        self.delta_ack = action[0] * 40 * math.pi / 180
        if self.delta_ack > 40*math.pi/180:
            self.delta_ack = 40*math.pi/180
        elif self.delta_ack < -40*math.pi/180:
            self.delta_ack = -40*math.pi/180

        deltaLeft=math.atan(self.car_l/(-self.car_d + self.car_l/math.tan(self.delta_ack)))
        deltaRight=math.atan(self.car_l/(self.car_d + self.car_l/math.tan(self.delta_ack)))



        vrep.simxSetJointTargetVelocity(clientID=self.vrep_client_id,
                                        jointHandle=self.left_motor_handles,
                                         targetVelocity=wheel_speed,
                                        operationMode=vrep.simx_opmode_oneshot)
        
        vrep.simxSetJointTargetVelocity(clientID=self.vrep_client_id,
                                        jointHandle=self.right_motor_handles,
                                        targetVelocity=wheel_speed,
                                        operationMode=vrep.simx_opmode_oneshot)


        vrep.simxSetJointTargetPosition(clientID=self.vrep_client_id,
                                        jointHandle=self.steeringLeft_handles,
                                        targetPosition=deltaLeft,
                                        operationMode=vrep.simx_opmode_oneshot)
        
        vrep.simxSetJointTargetPosition(clientID=self.vrep_client_id,
                                        jointHandle=self.steeringRight_handles,
                                        targetPosition=deltaRight,
                                        operationMode=vrep.simx_opmode_oneshot)
   
    #이 곳에서 csv 파일을 읽어 path를 따 옵니다.
    def readCSV(self):
        path_list = []
        with open('goal.csv') as file:
            reader = csv.reader(file)
            for row in reader:
                path_list += [[float(row[0]),float(row[1])]]
        path_list = path_list[:120]
        return path_list


    # path에서 목표하는 한 점에 차가 일정 이상 가까워지면 목표하는 점을 다음 점으로 바꿉니다.
    def goal_distance_calculate(self):
        self.goal_number
        abs_goal_position = self.path_list[self.goal_number]
        
        abs_goal_position = np.asarray(abs_goal_position)[:2]
        if self.goal_number == 0:
            abs_goal_position2 = [0,0]
        else :
            abs_goal_position2 = np.asarray(self.path_list[self.goal_number-1])[:2]

        goal_direction = abs_goal_position - abs_goal_position2
        #print('gd:', goal_direction)
        theta = math.atan2(goal_direction[1], goal_direction[0])
        #print ('theta:',theta)
        tan = math.tan(theta)
        #print ('tan:', tan)
        
        abs_robot_position = self.current_car_position[:2]
        #print('robot:', abs_robot_position[:2])
        if tan == 0:
            distance = abs(abs_robot_position[0]-abs_goal_position[0])
        elif tan == math.inf:
            distance = abs(abs_robot_position[1]-abs_goal_position[1])
        else:
            arctan = -1/tan
            distance = abs(arctan*abs_robot_position[0]-abs_robot_position[1]-arctan*abs_goal_position[0]+abs_goal_position[1])/((-1/tan)**2+1)**0.5


        #print('distance:', distance)
    
        if distance < 1:
            self.goal_number = self.goal_number + 1

        #print("Goal number "+str(goal_number))


    # 이곳에서 차의 현재 진행방향과 목표지점 사이의 각도를 측정합니다.
    def angleCheck(self):
        self.goal_number
        abs_goal_position = self.path_list[self.goal_number]
        
        abs_goal_position = np.asarray(abs_goal_position)[:2]

        if self.goal_number == 0:
            abs_goal_position2 = [0,0]
        else :
            abs_goal_position2 = np.asarray(self.path_list[self.goal_number-1])[:2]

        goal_direction = abs_goal_position - abs_goal_position2

        self.current_car_position
        self.prev_car_position
        car_direction = (self.current_car_position[0] - self.prev_car_position[0], self.current_car_position[1] - self.prev_car_position[1])
        try:
            theta1 = math.atan2(car_direction[1], car_direction[0])
            theta2 = math.atan2(goal_direction[1], goal_direction[0])
        except:
            theta1 = 0
            theta2 = 0
        self.angle = theta1 - theta2
        if self.angle > math.pi:
            self.angle = 2*math.pi - self.angle
        if self.angle < -1*math.pi:
            self.angle = 2*math.pi + self.angle


        #print('angle:', self.angle)



    # 양 옆 혹은 앞쪽의 lidar 거리가 일정 이상으로 가까워지면 멈추게 하고 -리워드를 부여합니다. 원하는 바퀴수를 넘어서 달리면 멈추게 합니다.
    def finishCheck(self):


        if self.front_lidar < 2.4:
            self.reset_reward = -4
            return True
        elif self.side_lidar < 1:
            self.reset_reward = -4
            return True
        elif self.goal_number ==119:
            return True
        else:
            self.reset_reward = 0
            return False
 



    # 속도를 올리기 위해 스트리밍을 사용한 부분입니다.
    def streaming(self):
        vrep.simxGetObjectOrientation(clientID=self.vrep_client_id,
                                                    objectHandle=self.car_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.vrep_client_id,self.car_handle,-1,vrep.simx_opmode_streaming)
        vrep.simxGetStringSignal(clientID=self.vrep_client_id, signalName='Team12_measurement', operationMode=vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(clientID=self.vrep_client_id, 
                                                    objectHandle=self.car_handle,
                                                    operationMode=vrep.simx_opmode_streaming)
        #vrep.simxStopSimulation(clientID=self.vrep_client_id, operationMode=vrep.simx_opmode_streaming)
        #vrep.simxStartSimulation(clientID=self.vrep_client_id, operationMode=vrep.simx_opmode_streaming)
        #vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_motorLeft', operationMode=vrep.simx_opmode_streaming)
        #vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_motorRight', operationMode=vrep.simx_opmode_streaming)
        #vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_steeringLeft', operationMode=vrep.simx_opmode_streaming)
        #vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='TEAM12_steeringRight', operationMode=vrep.simx_opmode_streaming)
        #vrep.simxGetObjectHandle(clientID=self.vrep_client_id,objectName='TEAM12_VehicleFrontPose', operationMode=vrep.simx_opmode_streaming)
    # 자동차의 현재 위치를 가져옵니다
    def Getcarposition(self):
        current_car_position = vrep.simxGetObjectPosition(self.vrep_client_id,self.car_handle,-1,vrep.simx_opmode_blocking)[1]
        current_car_position = current_car_position[:2]
        return current_car_position

