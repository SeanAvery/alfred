import os
from time import sleep
from os.path import dirname
from mujoco_py import load_model_from_path, MjSim, MjViewer, __file__

# load xml model from disc
tosser_model = load_model_from_path(dirname(dirname(__file__))  +"/xmls/tosser.xml")

sim = MjSim(tosser_model)
viewer = MjViewer(sim)

print(sim)
print(viewer)

sim_state = sim.get_state()

while True:
    sim.set_state(sim_state)

    for i in range(10000):
        if i < 500:
            sim.data.ctrl[:] = 0.0
        else:
            sim.dat.ctrl[:] = -1.0
        sleep(0.1)
        sim.step()
        viewer.render()

    if os.getenv('TESTING') is not None:
        break
