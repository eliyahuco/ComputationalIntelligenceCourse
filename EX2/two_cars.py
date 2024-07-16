import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time as t

class two_cars:
    m = 1145 # Kg
    dt = 0.001
    C = 0.4375/2
    road_Width = 4       # [m]
    road_Length = 100       # [m]
    road_Margin = 0.5        # [m]
    car_length = 2 #[m]
    collision_distance = car_length/2

    def __init__(self, y0 = 10):
        self.lead_car_func = lambda t: 3 * t + y0

        self.init_state()
        self.f = 0

        fig, self.ax = plt.subplots(figsize=(14,4))

    def init_state(self, x = [0, 0]):
        self.x = x[0]
        self.v = x[1]

        self.x_mem = np.zeros([0])
        self.v_mem = np.zeros([0])
        self.F_mem = np.zeros([0])
        self.x_lead_mem = np.zeros([0])

        self.t = 0.0

        self.x_lead = self.lead_car_func(self.t)
        self.distance = self.x_lead - self.x

    def step(self, f = 0):
        xd = self.v
        self.f = f #+ np.random.normal(0, 0.01)
        
        vd = 1/self.m * (f - self.C*self.v)

        self.x += self.dt*xd 
        self.v += self.dt*vd
        
        self.x_mem = np.append(self.x_mem, self.x)
        self.v_mem = np.append(self.v_mem, self.v)
        self.F_mem = np.append(self.F_mem, self.f)

        self.t += self.dt

        self.x_lead = self.lead_car_func(self.t)
        self.x_lead_mem = np.append(self.x_lead_mem, self.x_lead)

        self.distance = self.x_lead - self.x

        if self.distance < self.collision_distance:
            print('ERROR: Collision!')
            self.draw()
            plt.text(40, -1, 'Collision!', fontsize = 30, color = 'r')
            plt.show()
            return False
        
        return True

    def draw(self):
        ax = self.ax
        ax.clear()

        w = self.car_length
        ax.add_patch(mpatches.Rectangle((self.x, -w/4), w, w/2, facecolor='r', edgecolor='0.7'))
        ax.add_patch(mpatches.Rectangle((self.x_lead, -w/4), w, w/2, facecolor='y', edgecolor='0.7'))

        ax.plot([0, self.road_Length], [-self.road_Width/2, -self.road_Width/2], 'k--')
        ax.plot([0, self.road_Length], [self.road_Width/2, self.road_Width/2], 'k--')
        ax.plot([0, self.road_Length], [-self.road_Width/2-self.road_Margin, -self.road_Width/2-self.road_Margin], 'k-', linewidth=3)
        ax.plot([0, self.road_Length], [self.road_Width/2+self.road_Margin, self.road_Width/2+self.road_Margin], 'k-', linewidth=3)

        ax.set_xlim([0, self.road_Length])
        ax.set_ylim([-self.road_Width/2-self.road_Margin,self.road_Width/2+self.road_Margin])

        ax.set_aspect('equal')
        ax.set_title('Time: ' + str(round(self.t, 2)) + 'sec, distance between cars: ' + str(round(self.distance, 2)) + 'm')
        plt.pause(0.001)


if __name__ == '__main__':
    y0_input = 10
    IP = two_cars(y0 = y0_input)

    Tf = 60
    i = 0
    while IP.x < IP.road_Length and IP.t < Tf: # While car 1 did not reach end of road and time has not reached maximum limit Tf
        print('Time: ', round(IP.t, 2), 'sec, Position: ', round(IP.x, 2), 'm, distance: ', round(IP.distance, 2), 'm')
        success = IP.step(f = 1000) # Advance time by one step of dt seconds

        if not success:
            break

        if i % 50 == 0:
            IP.draw()
            


        i += 1

    
    
