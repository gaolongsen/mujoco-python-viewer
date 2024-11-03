import mujoco
import glfw
import numpy as np
import time
import pathlib
import yaml
import threading
from .callbacks import Callbacks  # Ensure this is correctly imported

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split('.')))


class MujocoViewer(Callbacks):
    def __init__(
            self,
            model,
            data,
            panel_num,
            mode='window',
            title="mujoco-python-viewer",
            width=None,
            height=None,
            window_start_x_pixel_offset=6,
            window_start_y_pixel_offset=30,
            hide_menus=False):
        super().__init__(hide_menus)
        if hide_menus:
            self._hide_graph = True

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ['offscreen', 'window']:
            raise NotImplementedError(
                "Invalid mode. Only 'offscreen' and 'window' are supported.")

        # Initialize control flags with default values
        self._run_speed = 1.0
        self._render_every_frame = True
        self._paused = False
        self._num = panel_num
        self._advance_by_one_step = False
        self._image_idx = 0
        self._image_path = "frame_%d.png"
        self._contacts = False
        self._joints = False
        self._inertias = False
        self._com = False
        self._shadows = False
        self._transparent = False
        self._wire_frame = False
        self._convex_hull_rendering = False
        self._hide_menus = hide_menus
        self._hide_graph = getattr(self, '_hide_graph', False)
        self._time_per_render = 1.0
        self._loop_count = 0

        # Keep true while running
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.home() / ".config/mujoco_viewer/config.yaml"

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        # Create GLFW window (initially hidden if offscreen)
        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)
        self.window = glfw.create_window(
            width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")
        glfw.make_context_current(self.window)
        glfw.set_window_pos(self.window,
                            window_start_x_pixel_offset,
                            window_start_y_pixel_offset)

        # Show window if in 'window' mode
        if self.render_mode == 'window':
            glfw.show_window(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)

        # Install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width / window_width

            # Override the key callback to handle space key for pausing
            glfw.set_key_callback(
                self.window, self._key_callback)

            # Set mouse-related callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_window_size_callback(self.window, self._window_size_callback)

        # Create MuJoCo visualization components
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()

        # Initialize two figures for plotting
        self.fig_bottom = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig_bottom)
        self.fig_bottom.flg_extend = 1
        self.fig_bottom.flg_symmetric = 0

        self.fig_center = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(self.fig_center)
        self.fig_center.flg_extend = 1
        self.fig_center.flg_symmetric = 0

        # Maximum number of points per line
        self._num_pnts = 1000

        # Initialize lines for both figures
        self._data_graph_line_names_bottom = []
        self._line_datas_bottom = []

        self._data_graph_line_names_center = []
        self._line_datas_center = []

        for n in range(mujoco.mjMAXLINE):
            for i in range(self._num_pnts):
                self.fig_bottom.linedata[n][2 * i] = float(-i)
                self.fig_center.linedata[n][2 * i] = float(-i)

        # Create MuJoCo rendering context
        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # Define viewports for both graphs
        width, height = framebuffer_width, framebuffer_height
        width_adjustment = width % 4

        # Right-Bottom Panel
        self.graph_viewport_bottom = mujoco.MjrRect(
            left=int(3 * width / 4) + width_adjustment,
            bottom=0,
            width=int(width / 4),
            height=int(height / 4),
        )

        # Right-Center Panel
        self.graph_viewport_center = mujoco.MjrRect(
            left=int(3 * width / 4) + width_adjustment,
            bottom=int(height / 4),
            width=int(width / 4),
            height=int(height / 4),
        )

        # Enable autorange for both figures
        self.axis_autorange(fig_idx=0)
        self.axis_autorange(fig_idx=1)

        # Load camera configuration (if available)
        pathlib.Path(self.CONFIG_PATH.parent).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(self.CONFIG_PATH).touch(exist_ok=True)
        with open(self.CONFIG_PATH, "r") as f:
            try:
                load_config = yaml.safe_load(f)
                if isinstance(load_config, dict):
                    for key, val in load_config.items():
                        if key in ["type", "fixedcamid", "trackbodyid", "lookat",
                                   "distance", "azimuth", "elevation"]:
                            setattr(self.cam, key, val)
                # Validate camera settings
                if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
                    if self.cam.fixedcamid >= self.model.ncam:
                        self.cam.fixedcamid = 0  # Default to first camera
                elif self.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    if self.cam.trackbodyid >= self.model.nbody:
                        self.cam.trackbodyid = 0  # Default to first body
            except yaml.YAMLError as e:
                print(f"Error loading camera config: {e}")

        # Get main viewport
        self.viewport = mujoco.MjrRect(
            left=0, bottom=0, width=framebuffer_width, height=framebuffer_height)

        # Overlay and markers
        self._overlay = {}
        self._markers = []

        # Initialize threading lock for GUI operations
        self._gui_lock = threading.Lock()

    # --------- Plot Management Methods --------- #

    def set_grid_divisions(self, x_div: int, y_div: int, x_axis_time: float = 0.0, fig_idx=0, override=False):
        """
        Set grid divisions for a specified graph panel.

        :param x_div: Number of divisions along the x-axis.
        :param y_div: Number of divisions along the y-axis.
        :param x_axis_time: Total time span represented on the x-axis.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        :param override: If True, override the x_axis_time restrictions.
        """
        if not override:
            assert x_axis_time >= self.model.opt.timestep * 50, (
                "Set [x_axis_time] >= [self.model.opt.timestep * 50] "
                "to ensure a suitable sampling rate."
            )
        if fig_idx == 0:
            fig = self.fig_bottom
        elif fig_idx == 1:
            fig = self.fig_center
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

        fig.gridsize[0] = x_div + 1
        fig.gridsize[1] = y_div + 1

        if x_axis_time != 0.0:
            self._num_pnts = int(x_axis_time / self.model.opt.timestep)
            print("self._num_pnts:", self._num_pnts)
            if self._num_pnts > 1000:
                self._num_pnts = 1000
                new_x_axis_time = self.model.opt.timestep * self._num_pnts
                print(
                    f"Maximum x_axis_time is: {new_x_axis_time} seconds. "
                    "Consider reducing x_axis_time or increasing timestep."
                )
            assert 1 <= self._num_pnts <= 1000, (
                "num_pnts should be between 1 and 1000, "
                f"currently: {self._num_pnts}"
            )
            self._time_per_div = (self.model.opt.timestep * self._num_pnts) / x_div
            self.set_x_label(
                xname=f"time/div: {self._time_per_div:.2f}s, total: {self.model.opt.timestep * self._num_pnts:.2f}s",
                fig_idx=fig_idx
            )

    def axis_autorange(self, fig_idx=0):
        """
        Enable autorange for a specified graph panel.

        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        if fig_idx == 0:
            fig = self.fig_bottom
        elif fig_idx == 1:
            fig = self.fig_center
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

        fig.range[0][0] = 1.0
        fig.range[0][1] = -1.0
        fig.range[1][0] = 1.0
        fig.range[1][1] = -1.0

    def set_graph_name(self, name: str, fig_idx=0):
        """
        Set the title of a specified graph panel.

        :param name: Title of the graph.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        assert isinstance(name, str), "Graph name must be a string."
        if fig_idx == 0:
            self.fig_bottom.title = name
        elif fig_idx == 1:
            self.fig_center.title = name
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def show_graph_legend(self, show_legend: bool = True, fig_idx=0):
        """
        Show or hide the legend of a specified graph panel.

        :param show_legend: Boolean to show or hide the legend.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        if show_legend:
            if fig_idx == 0:
                for i, name in enumerate(self._data_graph_line_names_bottom):
                    self.fig_bottom.linename[i] = name.encode('utf8')
                self.fig_bottom.flg_legend = True
            if fig_idx == 1:
                for i, name in enumerate(self._data_graph_line_names_center):
                    self.fig_center.linename[i] = name.encode('utf8')
                self.fig_center.flg_legend = True
            if fig_idx != 0 and fig_idx != 1:
                raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")
        else:
            if fig_idx == 0:
                self.fig_bottom.flg_legend = False
            if fig_idx == 1:
                self.fig_center.flg_legend = False
            if fig_idx != 0 and fig_idx != 1:
                raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def set_x_label(self, xname: str, fig_idx=0):
        """
        Set the x-axis label of a specified graph panel.

        :param xname: Label for the x-axis.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        assert isinstance(xname, str), "xname must be a string."
        if fig_idx == 0:
            self.fig_bottom.xlabel = xname
        elif fig_idx == 1:
            self.fig_center.xlabel = xname
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def add_graph_line(self, line_name, line_data=0.0, fig_idx=0):
        """
        Add a new line to a specified graph panel.

        :param line_name: Name of the line.
        :param line_data: Initial data value for the line.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        assert isinstance(line_name, str), "Line name must be a string."
        if fig_idx == 0:
            if line_name in self._data_graph_line_names_bottom:
                print(f"Line '{line_name}' already exists in bottom-right graph.")
                return
            if len(self._data_graph_line_names_bottom) >= mujoco.mjMAXLINE:
                print("Maximum number of lines reached for bottom-right graph.")
                return
            self._data_graph_line_names_bottom.append(line_name)
            self._line_datas_bottom.append(line_data)
        elif fig_idx == 1:
            if line_name in self._data_graph_line_names_center:
                print(f"Line '{line_name}' already exists in center-right graph.")
                return
            if len(self._data_graph_line_names_center) >= mujoco.mjMAXLINE:
                print("Maximum number of lines reached for center-right graph.")
                return
            self._data_graph_line_names_center.append(line_name)
            self._line_datas_center.append(line_data)
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def update_graph_line(self, line_name, line_data, fig_idx=0):
        """
        Update the data of an existing line in a specified graph panel.

        :param line_name: Name of the line to update.
        :param line_data: New data value for the line.
        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        if fig_idx == 0:
            if line_name in self._data_graph_line_names_bottom:
                idx = self._data_graph_line_names_bottom.index(line_name)
                self._line_datas_bottom[idx] = line_data
            else:
                raise NameError(
                    f"Line '{line_name}' not found in bottom-right graph. Add it before updating."
                )
        elif fig_idx == 1:
            if line_name in self._data_graph_line_names_center:
                idx = self._data_graph_line_names_center.index(line_name)
                self._line_datas_center[idx] = line_data
            else:
                raise NameError(
                    f"Line '{line_name}' not found in center-right graph. Add it before updating."
                )
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def sensorupdate(self):
        """
        Update sensor data for both plotting panels.
        """
        self.sensorupdate_bottom()
        self.sensorupdate_center()

    def sensorupdate_bottom(self):
        """
        Update sensor data for the bottom-right graph panel.
        """
        if self._paused:
            return  # Do not update data when paused

        pnt = int(mujoco.mju_min(self._num_pnts, self.fig_bottom.linepnt[0] + 1))

        for n in range(len(self._line_datas_bottom)):
            for i in range(pnt - 1, 0, -1):
                self.fig_bottom.linedata[n][2 * i + 1] = self.fig_bottom.linedata[n][2 * i - 1]
            self.fig_bottom.linepnt[n] = pnt
            self.fig_bottom.linedata[n][1] = self._line_datas_bottom[n]

    def sensorupdate_center(self):
        """
        Update sensor data for the center-right graph panel.
        """
        if self._paused:
            return  # Do not update data when paused

        pnt = int(mujoco.mju_min(self._num_pnts, self.fig_center.linepnt[0] + 1))

        for n in range(len(self._line_datas_center)):
            for i in range(pnt - 1, 0, -1):
                self.fig_center.linedata[n][2 * i + 1] = self.fig_center.linedata[n][2 * i - 1]
            self.fig_center.linepnt[n] = pnt
            self.fig_center.linedata[n][1] = self._line_datas_center[n]

    def update_graph_size(self, fig_idx=0):
        """
        Adjust the viewport size and position for a specified graph panel.

        :param fig_idx: Index of the figure (0 for bottom-right, 1 for center-right).
        """
        if fig_idx == 0:
            self.update_graph_size_bottom()
        elif fig_idx == 1:
            self.update_graph_size_center()
        else:
            raise IndexError("fig_idx must be 0 (bottom-right) or 1 (center-right).")

    def update_graph_size_bottom(self):
        """
        Adjust the viewport size and position for the bottom-right graph panel.
        """
        width, height = glfw.get_framebuffer_size(self.window)
        width_adjustment = width % 4
        self.graph_viewport_bottom.left = int(3 * width / 4) + width_adjustment
        self.graph_viewport_bottom.bottom = 0
        self.graph_viewport_bottom.width = int(width / 4)
        self.graph_viewport_bottom.height = int(height / 4)

    def update_graph_size_center(self):
        """
        Adjust the viewport size and position for the center-right graph panel.
        """
        width, height = glfw.get_framebuffer_size(self.window)
        width_adjustment = width % 4
        self.graph_viewport_center.left = int(3 * width / 4) + width_adjustment
        self.graph_viewport_center.bottom = int(height / 4)
        self.graph_viewport_center.width = int(width / 4)
        self.graph_viewport_center.height = int(height / 4)

    # --------- Marker and Overlay Methods --------- #

    def show_actuator_forces(
            self,
            f_render_list,
            rgba_list=[1, 0, 1, 1],
            force_scale=0.05,
            arrow_radius=0.03,
            show_force_labels=False,
    ) -> None:
        """
        Display actuator forces as arrows in the simulation view.

        :param f_render_list: List of lists containing joint name, actuator name, and label.
                              Example:
                              [
                                  ["jnt_name1", "act_name_1", "label1"],
                                  ["jnt_name2", "act_name_2", "label2"]
                              ]
        :param rgba_list: List defining the color and transparency of the arrows.
        :param force_scale: Scaling factor for the force magnitude.
        :param arrow_radius: Radius of the arrow.
        :param show_force_labels: Whether to display labels with force values.
        """
        if not isinstance(f_render_list, list):
            raise TypeError("f_render_list must be a list of lists.")
        for render_item in f_render_list:
            if not isinstance(render_item, list) or len(render_item) != 3:
                raise ValueError("Each item in f_render_list must be a list of three elements.")
            jnt_name, act_name, label = render_item
            force_value = self.data.actuator(act_name).force[0]
            if not show_force_labels:
                label_str = ""
            else:
                label_str = f"{label}: {force_value:.2f}"
            self.add_marker(
                pos=self.data.joint(jnt_name).xanchor,
                mat=self.rotation_matrix_from_vectors(
                    vec1=[0.0, 0.0, 1.0],
                    vec2=self.data.joint(jnt_name).xaxis
                ),
                size=[
                    arrow_radius,
                    arrow_radius,
                    force_value * force_scale,
                ],
                rgba=rgba_list,
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                label=label_str.encode('utf8') if label_str else b""
            )

    def add_marker(self, **marker_params):
        """
        Add a marker with specified parameters to be rendered in the scene.

        :param marker_params: Keyword arguments defining marker properties.
        """
        if 'label' in marker_params and isinstance(marker_params['label'], str):
            marker_params['label'] = marker_params['label'].encode('utf8')
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        """
        Internal method to add a marker to the MuJoCo scene.

        :param marker: Dictionary containing marker properties.
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                f'Ran out of geoms. maxgeom: {self.scn.maxgeom}'
            )

        g = self.scn.geoms[self.scn.ngeom]
        # Default values
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, bytes):
                setattr(g, key, value)
            else:
                raise ValueError(
                    f"Invalid type for attribute '{key}': {type(value)}"
                )

        self.scn.ngeom += 1

    # --------- Overlay Methods --------- #

    def _create_overlay(self):
        """
        Create overlay text for the simulation view.
        """
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        # Populate overlay information
        run_speed = getattr(self, '_run_speed', 1.0)
        render_every_frame = getattr(self, '_render_every_frame', False)
        paused = getattr(self, '_paused', False)
        image_idx = getattr(self, '_image_idx', 0)
        image_path = getattr(self, '_image_path', "frame_%d.png")
        contacts = getattr(self, '_contacts', False)
        joints = getattr(self, '_joints', False)
        inertias = getattr(self, '_inertias', False)
        com = getattr(self, '_com', False)
        shadows = getattr(self, '_shadows', False)
        transparent = getattr(self, '_transparent', False)
        wire_frame = getattr(self, '_wire_frame', False)
        convex_hull_rendering = getattr(self, '_convex_hull_rendering', False)

        if render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(
                topleft,
                f"Run speed = {run_speed:.3f} x real time",
                "[S]lower, [F]aster"
            )
        add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if render_every_frame else "Off"
        )
        add_overlay(
            topleft, f"Switch camera (#cams = {self.model.ncam})",
            f"[Tab] (camera ID = {self.cam.fixedcamid})"
        )
        add_overlay(
            topleft,
            "[C]ontact forces",
            "On" if contacts else "Off"
        )
        add_overlay(
            topleft,
            "[J]oints",
            "On" if joints else "Off"
        )
        add_overlay(
            topleft,
            "[G]raph Viewer",
            "Off" if self._hide_graph else "On"
        )
        add_overlay(
            topleft,
            "[I]nertia",
            "On" if inertias else "Off"
        )
        add_overlay(
            topleft,
            "Center of [M]ass",
            "On" if com else "Off"
        )
        add_overlay(
            topleft, "Shad[O]ws", "On" if shadows else "Off"
        )
        add_overlay(
            topleft,
            "T[r]ansparent",
            "On" if transparent else "Off"
        )
        add_overlay(
            topleft,
            "[W]ireframe",
            "On" if wire_frame else "Off"
        )
        add_overlay(
            topleft,
            "Con[V]ex Hull Rendering",
            "On" if convex_hull_rendering else "Off",
        )
        if paused is not None:
            if not paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[Right Arrow]"
                )
        add_overlay(topleft, "Toggle geomgroup visibility (0-5)",
                    ",".join(["On" if g else "Off" for g in self.vopt.geomgroup]))
        add_overlay(
            topleft,
            "Referenc[e] frames",
            mujoco.mjtFrame(self.vopt.frame).name
        )
        add_overlay(topleft, "[H]ide Menus", "")
        if image_idx > 0:
            fname = image_path % (image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", f"Saved as {fname}")
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")
        add_overlay(topleft, "[ESC] to Quit Application", "")
        add_overlay(topleft, "[BACKSPACE] to Reload Sim", "")

        # Bottom-left overlay
        fps = int(1 / getattr(self, '_time_per_render', 60))
        add_overlay(
            bottomleft, "FPS", f"{fps}"
        )
        if hasattr(self.data, 'solver_iter'):
            add_overlay(
                bottomleft, "Solver iterations", str(
                    self.data.solver_iter + 1)
            )
        else:
            add_overlay(
                bottomleft, "Solver iterations", "N/A"
            )
        step = int(round(self.data.time / self.model.opt.timestep))
        add_overlay(
            bottomleft, "Step", str(step)
        )
        add_overlay(bottomleft, "timestep", f"{self.model.opt.timestep:.5f}")

    # --------- Perturbation and Rendering Methods --------- #

    def apply_perturbations(self):
        """
        Apply any user-defined perturbations to the simulation.
        """
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None, depth=False):
        """
        Read pixel data from the simulation view.

        :param camid: Camera ID to capture pixels from. Use -1 for free camera.
        :param depth: Whether to capture depth information.
        :return: Tuple of (RGB image, Depth image) or just RGB image.
        """
        if self.render_mode == 'window':
            raise NotImplementedError(
                "Use 'render()' in 'window' mode."
            )

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window)
        # Update scene
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)
        # Render
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def render(self):
        """
        Render the simulation view along with the plotting panels.
        """
        if self.render_mode == 'offscreen':
            raise NotImplementedError(
                "Use 'read_pixels()' for 'offscreen' mode."
            )
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render."
            )
        if glfw.window_should_close(self.window):
            self.close()
            return

        # Define the update function to encapsulate rendering logic
        def update():
            # Fill overlay items on the top-left side
            self._create_overlay()

            render_start = time.time()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window)
            with self._gui_lock:
                # Update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # Add marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # Render main scene
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # Render overlay items
                for gridpos, (t1, t2) in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx
                    )

                # Handle rendering of both figures
                if not self._hide_graph:
                    # Render bottom-right graph
                    self.update_graph_size_bottom()
                    if not self._paused:
                        self.sensorupdate_bottom()
                    if self._num > 0:
                        mujoco.mjr_figure(
                            self.graph_viewport_bottom,
                            self.fig_bottom,
                            self.ctx
                        )

                    # Render center-right graph
                    self.update_graph_size_center()
                    if not self._paused:
                        self.sensorupdate_center()
                    if self._num > 1:
                        mujoco.mjr_figure(
                            self.graph_viewport_center,
                            self.fig_center,
                            self.ctx
                        )

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                                    0.1 * (time.time() - render_start)

            # Clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # Calculate loop count based on run speed and timestep
            loop_increment = self.model.opt.timestep / (
                    self._time_per_render * self._run_speed
            )
            self._loop_count += loop_increment
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # Clear markers
        self._markers[:] = []

        # Apply perturbations
        self.apply_perturbations()

    def close(self):
        """
        Close the GLFW window and terminate the context.
        """
        self.is_alive = False
        glfw.destroy_window(self.window)
        glfw.terminate()
        self.ctx.free()

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """
        Find the rotation matrix that aligns vec1 to vec2.

        :param vec1: A 3D "source" vector.
        :param vec2: A 3D "destination" vector.
        :return: A rotation matrix (3x3) that aligns vec1 with vec2.
        """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.isclose(c, -1.0):
            # 180 degrees rotation, choose arbitrary orthogonal vector
            orthogonal = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
            v = np.cross(a, orthogonal)
            v /= np.linalg.norm(v)
            rotation_matrix = -np.eye(3) + 2 * np.outer(v, v)
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    # --------- Key Callback Method --------- #

    def _key_callback(self, window, key, scancode, action, mods):
        """
        Handle key press events.

        :param window: The window that received the event.
        :param key: The keyboard key that was pressed or released.
        :param scancode: The system-specific scancode of the key.
        :param action: GLFW_PRESS, GLFW_RELEASE or GLFW_REPEAT.
        :param mods: Bit field describing which modifier keys were held down.
        """
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self._paused = not self._paused
                state = "paused" if self._paused else "running"
                print(f"Simulation {state}.")
            elif key == glfw.KEY_ESCAPE:
                self.close()
            # Add other key handlers as needed
            elif key == glfw.KEY_TAB:
                # Example: Switch camera
                self.switch_camera()
            # Implement other key functionalities as needed

    def switch_camera(self):
        """
        Switch to the next available camera.
        """
        if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            self.cam.fixedcamid = (self.cam.fixedcamid + 1) % self.model.ncam
            print(f"Switched to camera ID: {self.cam.fixedcamid}")
        elif self.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            self.cam.trackbodyid = (self.cam.trackbodyid + 1) % self.model.nbody
            print(f"Tracking body ID: {self.cam.trackbodyid}")
        else:
            # Handle free camera or other types if necessary
            pass

    # --------- Mouse Callback Methods --------- #

    def _cursor_pos_callback(self, window, xpos, ypos):
        """
        Handle cursor position events.

        :param window: The window that received the event.
        :param xpos: The new cursor x-coordinate, in pixels.
        :param ypos: The new cursor y-coordinate, in pixels.
        """
        # Pass the event to the base class or handle it here
        super()._cursor_pos_callback(window, xpos, ypos)

    def _mouse_button_callback(self, window, button, action, mods):
        """
        Handle mouse button events.

        :param window: The window that received the event.
        :param button: The mouse button that was pressed or released.
        :param action: GLFW_PRESS or GLFW_RELEASE.
        :param mods: Bit field describing which modifier keys were held down.
        """
        # Pass the event to the base class or handle it here
        super()._mouse_button_callback(window, button, action, mods)

    def _scroll_callback(self, window, xoffset, yoffset):
        """
        Handle scroll events.

        :param window: The window that received the event.
        :param xoffset: Scroll offset along the x-axis.
        :param yoffset: Scroll offset along the y-axis.
        """
        # Pass the event to the base class or handle it here
        super()._scroll_callback(window, xoffset, yoffset)

    def _window_size_callback(self, window, width, height):
        """
        Handle window resize events.

        :param window: The window that was resized.
        :param width: The new width, in pixels, of the window.
        :param height: The new height, in pixels, of the window.
        """
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
        self.viewport.width = framebuffer_width
        self.viewport.height = framebuffer_height

        # Update graph viewports
        self.update_graph_size_bottom()
        self.update_graph_size_center()
