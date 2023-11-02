import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(100000):
    mujoco.mj_step(model, data)
    viewer.render()
    if not viewer.is_alive:
        break

# close
viewer.close()
