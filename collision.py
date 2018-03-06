from time import sleep
import math
from mujoco_py import load_model_from_xml, MjSim, MjViewer

XML_MODEL = '''
<?xml version="1.0" ?>
    <mujoco>
        <worldbody>
            <body name="bot" pos="0 0.3 1.2">
                <joint axis="1 0 0" damping="1" name="slide0" pos="0 0 0" type="slide"/>
                <joint axis="0 1 0" damping="1" name="slide1" pos="0 0 0" type="slide" />
                <joint axis="0 0 1" damping="0.1" name="slide2" pos="0 0 0" type="slide" />
                <geom mass="10" pos="0 0 0" rgba="1 0 0 1" size="0.15" type="sphere" />
            </body>
            <body mocap="true" name="mocap" pos="0.5 0.5 0.5">
                <geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.1 0.1 0.1" type="box"></geom>
			    <geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.2 0.2 0.05" type="box"></geom>
            </body>
            <body name="cylinder" pos="0.1 0.1 0.2">
                <geom mass="1" size="0.15 0.15" type="cylinder" />
                <joint axis="1 0 0" damping="1" name="cylinder:slidex" pos="0 0 0" type="slide" />
                <joint axis="0 1 0" damping="1" name="cylinder:slidey" pos="0 0 0" type="slide" />
            </body>
            <body name="box" pos="-0.8 0 0.2">
                <geom mass="0.1" size="0.15 0.15 0.15" type="box" />
            </body>
            <body name="floor" pos="0 0 0.025">
                <geom condim="3" size="2.0 2.0 0.02" rgba="0 1 0 1" type="box" />
            </body>
        </worldbody>
        <actuator>
            <motor gear="2000.0" joint="slide0" />
            <motor gear="1000.0" joint="slide1" />
        </actuator>
        <actuator>
            <motor gear="2000.0" joint="cylinder:slidex" />
            <motor gear="1000.0" joint="cylinder:slidey" />
        </actuator>
    </mujoco>
'''

# setup mujuco landscape
model = load_model_from_xml(XML_MODEL)
sim = MjSim(model)
viewer = MjViewer(sim)

# simulation tick
t = 0

# run simultation
while True:
    sim.data.ctrl[0] = math.cos(t / 5) * 0.01
    sim.data.ctrl[1] = math.cos(t / 5 )* 0.01
    t+=1
    sim.step()
    viewer.render()
    sleep(0.01)
    if t > 1000 and os.getenv('TESTING') is not None:
        break
