# Multi-functional Viewer for MuJoCo in Python

[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![Powered by Spatial Maths](https://raw.githubusercontent.com/petercorke/spatialmath-python/master/.github/svg/sm_powered.min.svg)](https://github.com/petercorke/spatialmath-python)[![PyPI version](https://badge.fury.io/py/roboticstoolbox-python.svg)](https://badge.fury.io/py/roboticstoolbox-python)
[![Anaconda version](https://anaconda.org/conda-forge/roboticstoolbox-python/badges/version.svg)](https://anaconda.org/conda-forge/roboticstoolbox-python)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/roboticstoolbox-python.svg)[![Build Status](https://github.com/petercorke/robotics-toolbox-python/workflows/Test/badge.svg?branch=master)](https://github.com/petercorke/robotics-toolbox-python/actions?query=workflow%3ATest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive renderer to use with the official Python bindings for MuJoCo.

Starting with version 2.1.2, MuJoCo comes with native Python bindings officially supported by the MuJoCo devs.  

If you have been a user of `mujoco-py`, you might be looking to migrate.  
Some pointers on migration are available [here](https://mujoco.readthedocs.io/en/latest/python.html#migration-notes-for-mujoco-py).

# Install
```sh
git clone https://github.com/gaolongsen/multi-panel_mujoco-pyviewer.git
```

```sh
cd multi-panel_mujoco-pyviewer
```

```sh
pip install -e .
```

# Usage
#### How to render in a window?
```py
import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
```

The render should pop up and the simulation should be running.  
Double-click on a geom and hold `Ctrl` to apply forces (using right mouse button) and torques (using left mouse button). This version we add the plot show on bottom-right side of the render windows as shown below:

![ezgif-2-6758c40cdf](https://github.com/JackTony123/picx-images-hosting/raw/master/exp1.7w6lrjlcu0.gif)



![](https://github.com/JackTony123/picx-images-hosting/raw/master/exp2.6pnaiy0eby.gif)

#### How to render offscreen?
```py
import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen')
mujoco.mj_forward(model, data)
img = viewer.read_pixels(camid=2)
## do something cool with img
```

# Optional Parameters

- `title`: set the title of the window, for example: `viewer = mujoco_viewer.MujocoViewer(model, data, title='My Demo')` (defaults to `mujoco-python-viewer`). 
- `width`: set the window width, for example: `viewer = mujoco_viewer.MujocoViewer(model, data, width=300)` (defaults to full screen's width). 
- `height`: set the window height, for example: `viewer = mujoco_viewer.MujocoViewer(model, data, height=300)` (defaults to full screen's height). 
- `hide_menus`: set whether the overlay menus and graph should be hidden or not (defaults to `False`).

## Update - 11/02/2024

Now I update the Lib and can make sure you can create more than one data panel to show your variables update on the right-hand side of your render window. This would help your project to show different types of the variables clearly through different panels, especially different variables with different value range during the process, for example, if you want to show the wrench applied by robot arm(6 by 1 vector) and the position tracking error(3 by 1) together through the panel.

For example, when you call `Mujoco` in you main code, you can initiated the parameter `panel_num` like `interface = Mujoco(robot_config, dt=0.001, panel_num=2)` to create another panel on the right-center part on your render window. 

![](https://github.com/JackTony123/picx-images-hosting/raw/master/double_panel.3d4svg0oda.png)

Note that if you just want only one data panel on the **right-bottom side** of the window, you don't need to initiate `panel_num` because its inital value is `1`;  Here is the example code that how to create two data panels on the render window:

```python
# Assume 'model' and 'data' are already defined MuJoCo objects
viewer = MujocoViewer(model, data, panel_num=2) # note that you need to initialize panel_num = 2

# Configure bottom-right graph (fig_idx=0)
viewer.set_graph_name("Joint Angles", fig_idx=0)
viewer.set_x_label("Time (s)", fig_idx=0)
viewer.show_graph_legend(True, fig_idx=0)
viewer.set_grid_divisions(x_div=10, y_div=5, x_axis_time=10.0, fig_idx=0)

# Add lines to bottom-right graph
viewer.add_graph_line("Joint1 Angle", line_data=0.0, fig_idx=0)
viewer.add_graph_line("Joint2 Angle", line_data=0.0, fig_idx=0)

# Configure center-right graph (fig_idx=1)
viewer.set_graph_name("Joint Velocities", fig_idx=1)
viewer.set_x_label("Time (s)", fig_idx=1)
viewer.show_graph_legend(True, fig_idx=1)
viewer.set_grid_divisions(x_div=10, y_div=5, x_axis_time=10.0, fig_idx=1)

# Add lines to center-right graph
viewer.add_graph_line("Joint1 Velocity", line_data=0.0, fig_idx=1)
viewer.add_graph_line("Joint2 Velocity", line_data=0.0, fig_idx=1)

# In your simulation loop, update the plots
while viewer.is_alive:
    # ... your simulation step ...

    # Example data retrieval (replace with actual data)
    joint1_angle = data.qpos[0]
    joint2_angle = data.qpos[1]
    joint1_velocity = data.qvel[0]
    joint2_velocity = data.qvel[1]

    # Update bottom-right graph
    viewer.update_graph_line("Joint1 Angle", joint1_angle, fig_idx=0)
    viewer.update_graph_line("Joint2 Angle", joint2_angle, fig_idx=0)

    # Update center-right graph
    viewer.update_graph_line("Joint1 Velocity", joint1_velocity, fig_idx=1)
    viewer.update_graph_line("Joint2 Velocity", joint2_velocity, fig_idx=1)

    # Render the viewer
    viewer.render()

```

Here is an example from one of our research work and you can see for the wrench applied from the end-effector of UR5e and the state variables from the hinge can be shown clearly on the right-center side and right-bottom side, respectively.

<img src="https://github.com/JackTony123/picx-images-hosting/raw/master/two_panels_demo.5fklkfd2wz.webp" style="zoom:45%;" />

Note that in order to show the legends on your multiple data panel successfully. You must make sure `show_graph_legend` function must be behind of all of your `add_graph_line` functions.

## Update - 05/07/2025

Add additional two data panels on the right-top side (use full space on the right-hand side) to show more data for more robotics system (because my current project needs at least 3 manipulators and 1 manipulated object so I created more panels for usage).

![](https://github.com/JackTony123/picx-images-hosting/raw/master/update_mj_viewer.6t7c0wtgnw.webp)

If you want to call the third panel, you just follow the similar operation as the right-bottom and right-center panel but set `fig_idx=2` in the `update_graph_line` function as the example below:

```python
    interface.viewer.update_graph_line(line_name="Force_X", line_data=torque_force_data3[:3][0], fig_idx=2)
    interface.viewer.update_graph_line(line_name="Force_Y", line_data=torque_force_data3[:3][1], fig_idx=2)
    interface.viewer.update_graph_line(line_name="Force_Z", line_data=torque_force_data3[:3][2], fig_idx=2)
    interface.viewer.update_graph_line(line_name="Torque_X", line_data=torque_force_data3[:3][0], fig_idx=2)
    interface.viewer.update_graph_line(line_name="Torque_Y", line_data=torque_force_data3[3:][1], fig_idx=2)
    interface.viewer.update_graph_line(line_name="Torque_Z", line_data=torque_force_data3[3:][2], fig_idx=2)
```

if you want to add another data panel, you can set `fig_idx=3` in the  `update_graph_line` function.
