# Viewer for MuJoCo in Python

Interactive renderer to use with the official Python bindings for MuJoCo.

Starting with version 2.1.2, MuJoCo comes with native Python bindings officially supported by the MuJoCo devs.  

If you have been a user of `mujoco-py`, you might be looking to migrate.  
Some pointers on migration are available [here](https://mujoco.readthedocs.io/en/latest/python.html#migration-notes-for-mujoco-py).

# Install
```sh
git clone https://github.com/gaolongsen/mujoco-python-viewer.git
```

```sh
cd mujoco-python-viewer
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
