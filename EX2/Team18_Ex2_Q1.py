"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

***both authors have contributed equally to the assignment.***
we were working together on the assignment and we both participated in the writing of the code and the writing of the report
---------------------------------------------------------------------------------
Short Description:



---------------------------------------------------------------------------------
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from two_cars import two_cars

# Initial condition for Car 2
y0_input = 10

# Create the simulation instance
Cars = two_cars(y0=y0_input)

# Define fuzzy variables
distance = ctrl.Antecedent(np.arange(0, 31, 1), 'distance')
speed = ctrl.Antecedent(np.arange(0, 51, 1), 'speed')
force = ctrl.Consequent(np.arange(-1500, 3001, 1), 'force')

# Define membership functions for distance
distance['near'] = fuzz.trimf(distance.universe, [0, 0, 10])
distance['medium'] = fuzz.trimf(distance.universe, [5, 15, 25])
distance['far'] = fuzz.trimf(distance.universe, [20, 30, 30])

# Define membership functions for speed
speed['slow'] = fuzz.trimf(speed.universe, [0, 0, 25])
speed['medium'] = fuzz.trimf(speed.universe, [10, 25, 40])
speed['fast'] = fuzz.trimf(speed.universe, [30, 50, 50])

# Define membership functions for force
force['large_negative'] = fuzz.trimf(force.universe, [-1500, -1500, -500])
force['small_negative'] = fuzz.trimf(force.universe, [-1000, -500, 0])
force['zero'] = fuzz.trimf(force.universe, [-500, 0, 500])
force['small_positive'] = fuzz.trimf(force.universe, [0, 500, 1000])
force['large_positive'] = fuzz.trimf(force.universe, [500, 3000, 3000])

# Define fuzzy rules
rule1 = ctrl.Rule(distance['near'] & speed['fast'], force['large_negative'])
rule2 = ctrl.Rule(distance['near'] & speed['medium'], force['small_negative'])
rule3 = ctrl.Rule(distance['near'] & speed['slow'], force['zero'])
rule4 = ctrl.Rule(distance['medium'] & speed['fast'], force['small_negative'])
rule5 = ctrl.Rule(distance['medium'] & speed['medium'], force['zero'])
rule6 = ctrl.Rule(distance['medium'] & speed['slow'], force['small_positive'])
rule7 = ctrl.Rule(distance['far'] & speed['fast'], force['zero'])
rule8 = ctrl.Rule(distance['far'] & speed['medium'], force['small_positive'])
rule9 = ctrl.Rule(distance['far'] & speed['slow'], force['large_positive'])

# Create control system and simulation
force_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
force_sim = ctrl.ControlSystemSimulation(force_ctrl)

# Simulate the cars
positions_car1 = []
positions_car2 = []
distances = []
speeds = []
forces = []
times = []

for t in range(101):
    # Apply the force and update the state
    force_sim.input['distance'] = Cars.distance
    force_sim.input['speed'] = Cars.v
    force_sim.compute()

    f = force_sim.output['force']
    Cars.step(f)

    positions_car1.append(Cars.x)
    positions_car2.append(Cars.x_lead)
    distances.append(Cars.distance)
    speeds.append(Cars.v)
    forces.append(f)
    times.append(t)

    if Cars.x >= 50:
        break

# Plot results
plt.figure(figsize=(15, 10))

# Plot positions
plt.subplot(3, 2, 1)
plt.plot(times, positions_car1, label='Car 1')
plt.plot(times, positions_car2, label='Car 2')
plt.axvline(x=50, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Positions of Cars')

# Plot distances
plt.subplot(3, 2, 2)
plt.plot(times, distances)
plt.axvline(x=50, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.title('Distance between Cars')

# Plot speed of Car 1
plt.subplot(3, 2, 3)
plt.plot(times, speeds)
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Speed of Car 1')

# Plot force applied to Car 1
plt.subplot(3, 2, 4)
plt.plot(times, forces)
plt.axvline(x=50, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Force applied to Car 1')

# Plot membership functions
plt.subplot(3, 2, 5)
distance.view()

plt.subplot(3, 2, 6)
speed.view()

plt.tight_layout()
plt.show()
