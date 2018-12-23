import numpy as np
import gym
from gym import spaces
from api import vrep
from time import sleep
import csv
import math, time
import random
goal_number = 0
class New_Rules:
    
    def __init__(self, enables_acturator_dynamics = False):

        self.vrep_client_id=vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP
        
        self.current_goal = 0.

        self.path_list = []

        self.current_car_position = []
        while True:
            print('Connected to remote API server')
            if self.vrep_client_id != -1:
                break
            else:
                sleep(0.2)
        
        self.path_list = self.readCSV()
        
        self.left_motor_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Team12_motorLeft', operationMode=vrep.simx_opmode_blocking)[1]
        
        self.right_motor_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Team12_motorRight', operationMode=vrep.simx_opmode_blocking)[1]
        
        self.steeringLeft_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Team12_steeringLeft', operationMode=vrep.simx_opmode_blocking)[1]
        
        self.steeringRight_handles = vrep.simxGetObjectHandle(clientID=self.vrep_client_id, objectName='Team12_steeringRight', operationMode=vrep.simx_opmode_blocking)[1]

        self.car_handle = vrep.simxGetObjectHandle(clientID=self.vrep_client_id,objectName='Team12_VehicleFrontPose', operationMode=vrep.simx_opmode_blocking)[1]

        self.collision_handle = vrep.simxGetCollisionHandle(clientID=self.vrep_client_id,collisionObjectName='Collision', operationMode=vrep.simx_opmode_blocking)[1]

        self.goal_number = 0

        self.goal_angle = 0

    def streaming(self):
        
        vrep.simxGetObjectOrientation(clientID=self.vrep_client_id,
                                                    objectHandle=self.car_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.vrep_client_id,self.car_handle,-1,vrep.simx_opmode_streaming)
        #vrep.simxGetObjectPosition(self.vrep_client_id,self.goal_handle,-1,vrep.simx_opmode_streaming)
        vrep.simxGetStringSignal(clientID=self.vrep_client_id, signalName='Team12_measurement', operationMode=vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(clientID=self.vrep_client_id, 
                                                    objectHandle=self.car_handle,
                                                    operationMode=vrep.simx_opmode_streaming)
     

    def car_velocity(self):

        velo = vrep.simxGetObjectOrientation(clientID=self.vrep_client_id,
                                                    objectHandle=self.car_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_buffer)[1]
        
        self.current_car_velocity = (velo[0]**2+velo[1]**2)**0.5
        return velo[:2]



    def angleCheck2(self):
	    velocity_cos = self.angleDif
	    velocity_sin = math.sin(velocity_cos)

	    return velocity_sin

    #우선 현재는 lidardata2 가 왼쪽을, 1이 오른쪽을 감지하고 있다고 보시면 됩니다
    def getLidarData(self):
        _, lidar_data_bin = vrep.simxGetStringSignal(clientID=self.vrep_client_id, signalName='Team12_measurement', operationMode=vrep.simx_opmode_blocking)  
        self.lidar_data = np.array(vrep.simxUnpackFloats(lidar_data_bin), dtype=float)
        
        

        return self.lidar_data


    def getMinLidarData(self):        
        self.lidar_data = self.getLidarData()
        return np.min(self.lidar_data)



    def setMotorSpeeds(self, action):

        L = 2.2
        w = 1.44     # distance between left and right wheel
        d = 0.63407

        
        delta_ack = action[0]
        vel = action[1]
        wheel_speed = 2*vel/d
        R = L/np.abs(delta_ack)
        delta_o = np.arctan2(L,R+w/2)
        delta_i = np.arctan2(L,R-w/2)
        if delta_ack > 0:
            deltaLeft = delta_i
            deltaRight = delta_o
        elif delta_ack < 0:
            deltaLeft = -delta_o
            deltaRight = -delta_i
        else:
            deltaLeft = 0
            deltaRight = 0


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
   
    def readCSV(self):
        path_list = []
        with open('goal.csv') as file:
            reader = csv.reader(file)
            #for i in range(3):
            for row in reader:
                path_list += [[float(row[0]),float(row[1])]]
        path_list += path_list
        path_list += path_list
        print(path_list)
        return path_list

    def goal_distance_calculate(self):
        global goal_number

        abs_goal_position = self.path_list[goal_number]
        
        abs_goal_position = np.asarray(abs_goal_position)[:2]
        if goal_number == 0:
            abs_goal_position2 = [0,0]
        else :
            abs_goal_position2 = np.asarray(self.path_list[goal_number-1])[:2]
        #print ('goal position:', abs_goal_position)
        #print('pre_goal:', abs_goal_position2)

        goal_direction = abs_goal_position - abs_goal_position2
        #print('gd:', goal_direction)
        tan = goal_direction[1]/goal_direction[0]
            
        #print ('tan:', tan)
        
        self.abs_robot_position = vrep.simxGetObjectPosition(clientID=self.vrep_client_id, 
                                                    objectHandle=self.car_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_buffer)[1]
        if tan == 0:
            distance = abs(self.abs_robot_position[0]-abs_goal_position[0])
        elif tan == math.inf:
            distance = abs(self.abs_robot_position[1]-abs_goal_position[1])
        else:
            arctan = -1/tan
            distance = abs(arctan*self.abs_robot_position[0]-self.abs_robot_position[1]-arctan*abs_goal_position[0]+abs_goal_position[1])/((-1/tan)**2+1)**0.5
        #print ('distance:', distance)


    
        if distance < 2:
            goal_number = goal_number + 1

        print("Goal number "+str(goal_number))

        return distance


    def distance_to_centerline(self):
        global goal_number
        abs_goal_position = self.path_list[goal_number]
        #print('distance to center line!!')
        abs_goal_position = np.asarray(abs_goal_position)[:2]
        if goal_number == 0:
            abs_goal_position2 = [0,0]
        else :
            abs_goal_position2 = np.asarray(self.path_list[goal_number-1])[:2]
        #print ('goal position:', abs_goal_position)
        #print('pre_goal:', abs_goal_position2)

        goal_direction = abs_goal_position - abs_goal_position2
        #print('gd:', goal_direction)
        tan = goal_direction[1]/goal_direction[0]
        #print ('tan:', tan)
        self.abs_robot_position = vrep.simxGetObjectPosition(clientID=self.vrep_client_id, 
                                                    objectHandle=self.car_handle,
                                                    relativeToObjectHandle=-1,
                                                    operationMode=vrep.simx_opmode_buffer)[1]
        print ('car position:', self.abs_robot_position)
        if tan == 0:
            distance2 = abs(self.abs_robot_position[1]-abs_goal_position[1])
        elif tan == math.inf:
            distance2 = abs(self.abs_robot_position[0]-abs_goal_position[0])
        else:
            distance2 = abs(tan*self.abs_robot_position[0]-self.abs_robot_position[1]-tan*abs_goal_position[0]+abs_goal_position[1])/(tan**2+1)**0.5
        print ('distance2:', distance2)
        self.distance2 = distance2
        return self.distance2


    def distance_wall(self):
        
        self.distance3 = 4 - self.distance2

        return self.distance3


    #reward는 현재 vcos, vsin, 양쪽에서 나온 라이다 데이터로 구성되어 있고, 이 부분의 비율은 고치는 중입니다. + 다른 reward 값을 어떻게 줄지에 대해서도 고민중이구요.


# 위험 감지, 사전 회피
    def isDanger(self):
        self.getLidarData()
        In_danger_left = 0
        In_danger_right = 0
        # 차량 진행방향 중심선에서 좌우 4.5미터 이내의 거리에 물체가 있을 경우 각각 왼쪽, 오른쪽 위험 감지
        for i in range(450,540):
            theta = (180-i)*math.pi/180
            if self.lidar_data[i] < (4.5 / math.sin(theta)) and self.lidar_data[i] < 9:
                In_danger_left += 1
        for i in range(541,630):
            theta = (i-180)*math.pi/180
            if self.lidar_data[i] < (4.5 / math.sin(theta)) and self.lidar_data[i] < 9:
                In_danger_right += 1
        # 양쪽 모두에서 위험이 감지된 경우, 왼쪽과 오른쪽 30도 중에서 감지된 거리가 긴 쪽으로 이동(물체가 없는쪽으로 이동)
        if In_danger_left and In_danger_right:
            if self.lidar_data[150] < self.lidar_data[210]:
                return [0.7, 5]
            if self.lidar_data[150] > self.lidar_data[210]:
                return [-0.7, 5]
        # 왼쪽에서 위험이 감지된 경우 오른쪽으로 이동
        elif In_danger_left:
            return [0.7, 5]
        # 오른쪽에서 위험이 감지된 경우 왼쪽으로 이동
        elif In_danger_right:
            return [-0.7, 5]
        else :
            return [0, 7]

# 충돌 가능성 감지
    def collidable(self):
        # 라이다 데이터 수신
        self.getLidarData()
        collidable = False
        # 전방 좌우 20도, 6미터 이내에 물체가 있으면 충돌 위험
        if np.min(self.lidar_data[160:200]) < 6:
            collidable = True
        # 충돌위험 감지시
        if collidable:
            print('collide alert')
            # 최소거리가 먼 쪽으로 회피 기동
            if np.min(self.lidar_data[120:180]) < np.min(self.lidar_data[180:240]):
                self.avoid = [0.7, 6]
            elif np.min(self.lidar_data[120:180]) > np.min(self.lidar_data[180:240]):
                self.avoid = [-0.7, 6]
            # 최소거리가 같으면 직진
            else:
                self.avoid = [0,6]
            return True
        else :
            return False

# 후진
    def reverse(self):
        print('reverse')
        # 최소거리가 먼쪽으로 후진
        for i in range(40):
            self.getLidarData()
            if np.min(self.lidar_data[120:180]) < np.min(self.lidar_data[180:240]):
                self.setMotorSpeeds([-0.7,-8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            elif np.min(self.lidar_data[120:180]) > np.min(self.lidar_data[180:240]):
                self.setMotorSpeeds([0.7,-8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            else:
                self.setMotorSpeeds([0, -8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            self.goal_distance_calculate()
        # 최소거리가 먼쪽으로 회피
        for i in range(50):
            self.getLidarData()
            if np.min(self.lidar_data[120:180]) < np.min(self.lidar_data[180:240]):
                self.setMotorSpeeds([0.7, 8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            elif np.min(self.lidar_data[120:180]) > np.min(self.lidar_data[180:240]):
                self.setMotorSpeeds([-0.7, 8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            else:
                self.setMotorSpeeds([0, 8])
                vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            self.goal_distance_calculate()

    def angleCheck(self):
        # 골 방향 = 현재 goal 좌표 - 이전 goal 좌표
        if goal_number == 0:
            goal_direction = self.path_list[0]
        else :
            goal_direction = [self.path_list[goal_number][0] - self.path_list[goal_number-1][0], self.path_list[goal_number][1] - self.path_list[goal_number-1][1]]
        # 골 각도 : 절대 좌표계에서의 골방향의 각도
        goal_angle = math.atan2(goal_direction[1],goal_direction[0])
        # 차 방향 = 차의 속도 방향
        car_direction = self.car_velocity()
        # 차 각도 : 절대 좌표계에서의 차량의 방향의 각도
        car_angle = math.atan2(car_direction[1],car_direction[0])

        # 절대 좌표계에서 두 방향의 차이를 구한다. (양수일경우 차량이 골방향에 대해 왼쪽방향)
        self.angleDif = car_angle - goal_angle
        print('angle:', self.angleDif)
        return self.angleDif


# 잘못된 방향 감지
    def wrongDirection(self):
        # 차량 방향과 골방향의 각도가 90도 이상 차이나면
        if abs(self.angleDif) > math.pi/2 :
            # 잘못된 방향임을 알림
            return True
        else :
            return False

# 유턴
    def Uturn(self):
        # 제대로된 방향으로 갈 때 까지
        while self.wrongDirection():
            self.detection = vrep.simxReadCollision(clientID=self.vrep_client_id,collisionObjectHandle=self.collision_handle, operationMode=vrep.simx_opmode_buffer)[1]
            # 각도가 왼쪽으로 틀어졌을경우 오른쪽으로 회전, 오른쪽으로 틀어졌을 경우 왼쪽으로 회전
            if self.detection == False :
                if self.angleDif > 0 :
                    self.setMotorSpeeds([0.7, 4])
                    vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
                else :
                    self.setMotorSpeeds([-0.7, 4])
                    vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
            # 충돌시 reverse() 함수 호출 => 후진
            else :
                self.reverse()
            self.goal_distance_calculate()
            self.angleCheck()
                
# 메인 스텝
    def step(self):
        # 차량 속도 확인, 충돌확인
        self.car_velocity()
        self.detection = vrep.simxReadCollision(clientID=self.vrep_client_id,collisionObjectHandle=self.collision_handle, operationMode=vrep.simx_opmode_buffer)[1]
        # 골과 거리, 차량 방향과 차이 각도 확인
        self.goal_distance_calculate()
        self.angleCheck()
        # 잘못된 방향으로 가는경우
        if self.wrongDirection():
            # 유턴
            self.Uturn()
        # 충돌 가능성 감지
        elif self.collidable():
            #충돌여부 판단
            if self.current_car_velocity < 0.001 or self.detection == True :
                # 충돌시 후진
                self.reverse()
                return True
            # 충돌 안했을 경우 회피 기동
            self.setMotorSpeeds(self.avoid)
        #충돌 했을 경우
        elif self.detection == True :
            # 후진
            self.reverse()
        # 별일 없을 경우
        else :
            # 위험 가능성을 감지하며 이동
            self.setMotorSpeeds(self.isDanger())
        vrep.simxSynchronousTrigger(clientID=self.vrep_client_id)
        return True

# 리셋
    def reset(self):
        # 시물레이션 종료, 재시작
        vrep.simxSynchronous(self.vrep_client_id,True)
        print("ENV reset")
        vrep.simxStopSimulation(clientID=self.vrep_client_id,operationMode=vrep.simx_opmode_blocking)
        print('sim stopped')
        time.sleep(0.5)
        self.streaming()
        # 골넘버 초기화
        global goal_number
        goal_number = 0
        vrep.simxStartSimulation(clientID=self.vrep_client_id,operationMode=vrep.simx_opmode_blocking)
        print('sim started')
        time.sleep(0.2)
        # 골 거리 초기화
        self.goal_distance_calculate()
        vrep.simxSynchronousTrigger(self.vrep_client_id)
        vrep.simxSynchronousTrigger(self.vrep_client_id)
        vrep.simxSynchronousTrigger(self.vrep_client_id)
        vrep.simxSynchronousTrigger(self.vrep_client_id)
        time.sleep(1)
        # 라이다 데이터 수집
        self.getMinLidarData()

        
        return True


    def finishCheck(self):
        #l = obs[1]
        #d = self.goal_distance_calculate()
        #print ('d:', d)
        #pg = self.prev_goal_distance
        #print('pg', pg)

        #d = d - pg
        #minimum = np.min(self.minDatas)
        # if minimum < 1.:
        #     return True
        print('carvelocity 각각:', carvelocity)
        carvelocity = self.current_car_velocity
        self.detection = vrep.simxReadCollision(clientID=self.vrep_client_id,collisionObjectHandle=self.collision_handle, operationMode=vrep.simx_opmode_buffer)[1]

        if self.detection == True:
            # print('b')
            return True
        #elif carvelocity < 0.01:
        #    return True
        #if d < 0.1:
            # print('a')
        #    return True
        #elif self.getReward() < 0:
        #    print('no progress')
        #    return True
        else:
            return False
 