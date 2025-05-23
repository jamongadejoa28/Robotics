import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import koreanize_matplotlib # 한글 폰트 설정을 위해
import sys

# --- Constants ---
MAX_REACH_SUM = 10.0
WORKSPACE_LIMIT = 10.0
MIN_LINK_LENGTH = 0.1
DECIMAL_PLACES = 2
ANGLE_STEP = 1.0
LENGTH_STEP = 0.1
SINGULARITY_THRESHOLD_DEG = 5.0 # Degrees close to 0 or 180

# --- Kinematics Class ---
class RobotKinematics:
    def __init__(self, L1_init, L2_init, L3_init):
        self.L1 = L1_init
        self.L2 = L2_init
        self.L3 = L3_init
        self.q1_rad, self.q2_rad, self.q3_rad = 0.0, 0.0, 0.0 # Current angles in radians

    def set_link_lengths(self, L1, L2, L3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def set_joint_angles_rad(self, q1_rad, q2_rad, q3_rad):
        self.q1_rad = normalize_angle(q1_rad)
        self.q2_rad = normalize_angle(q2_rad)
        self.q3_rad = normalize_angle(q3_rad)

    def forward_kinematics(self):
        P0 = np.array([0.0, 0.0])
        abs_q1 = self.q1_rad
        P1_x = self.L1 * np.cos(abs_q1)
        P1_y = self.L1 * np.sin(abs_q1)
        P1 = np.array([P1_x, P1_y])

        abs_q2 = self.q1_rad + self.q2_rad
        P2_x = P1[0] + self.L2 * np.cos(abs_q2)
        P2_y = P1[1] + self.L2 * np.sin(abs_q2)
        P2 = np.array([P2_x, P2_y])

        abs_q3 = self.q1_rad + self.q2_rad + self.q3_rad
        P3_x = P2[0] + self.L3 * np.cos(abs_q3)
        P3_y = P2[1] + self.L3 * np.sin(abs_q3)
        P3 = np.array([P3_x, P3_y])
        return P0, P1, P2, P3

    def inverse_kinematics(self, target_x, target_y, phi_e_rad=0):
        xw = target_x - self.L3 * np.cos(phi_e_rad)
        yw = target_y - self.L3 * np.sin(phi_e_rad)
        D_sq = xw**2 + yw**2

        if self.L1 <= 1e-6 or self.L2 <= 1e-6: return None, None
        dist_xw_yw = np.sqrt(D_sq)

        if dist_xw_yw > self.L1 + self.L2 + 1e-6 or \
           dist_xw_yw < np.abs(self.L1 - self.L2) - 1e-6:
            return None, None

        if D_sq < 1e-9:
            if np.abs(self.L1 - self.L2) < 1e-9:
                q1_up_calc, q2_raw_up_calc = 0, np.pi
                q1_down_calc, q2_raw_down_calc = 0, np.pi
            else:
                return None, None
        else:
            cos_q2_raw_num = D_sq - self.L1**2 - self.L2**2
            cos_q2_raw_den = 2 * self.L1 * self.L2
            if np.abs(cos_q2_raw_den) < 1e-9: return None, None
            
            cos_q2_raw = np.clip(cos_q2_raw_num / cos_q2_raw_den, -1.0, 1.0)
            q2_raw_up_calc = -np.arccos(cos_q2_raw)
            q2_raw_down_calc = np.arccos(cos_q2_raw)

            k1_up = self.L1 + self.L2 * np.cos(q2_raw_up_calc)
            k2_up = self.L2 * np.sin(q2_raw_up_calc)
            q1_up_calc = np.arctan2(yw, xw) - np.arctan2(k2_up, k1_up)

            k1_down = self.L1 + self.L2 * np.cos(q2_raw_down_calc)
            k2_down = self.L2 * np.sin(q2_raw_down_calc)
            q1_down_calc = np.arctan2(yw, xw) - np.arctan2(k2_down, k1_down)

        q3_up_calc = phi_e_rad - (q1_up_calc + q2_raw_up_calc)
        q3_down_calc = phi_e_rad - (q1_down_calc + q2_raw_down_calc)

        sol1 = (normalize_angle(q1_up_calc), normalize_angle(q2_raw_up_calc), normalize_angle(q3_up_calc))
        sol2 = (normalize_angle(q1_down_calc), normalize_angle(q2_raw_down_calc), normalize_angle(q3_down_calc))
        return sol1, sol2

    def is_solution_near_singularity(self, sol_rad):
        if sol_rad is None: return False
        _, q2_r, q3_r = sol_rad
        thresh_rad = np.deg2rad(SINGULARITY_THRESHOLD_DEG)
        if abs(q2_r) < thresh_rad or abs(abs(q2_r) - np.pi) < thresh_rad: return True
        if abs(q3_r) < thresh_rad or abs(abs(q3_r) - np.pi) < thresh_rad: return True
        return False
    
    def check_current_angles_singularity(self):
        is_singular = False
        messages = []
        thresh_rad = np.deg2rad(SINGULARITY_THRESHOLD_DEG)
        if abs(self.q2_rad) < thresh_rad or abs(abs(self.q2_rad) - np.pi) < thresh_rad:
            is_singular = True
            messages.append("팔꿈치 특이점")
        if abs(self.q3_rad) < thresh_rad or abs(abs(self.q3_rad) - np.pi) < thresh_rad:
            is_singular = True
            messages.append("손목 특이점")
        return is_singular, messages


def normalize_angle(angle_rad):
    angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    return angle_rad

# --- GUI Application Class ---
class RobotArmGUI:
    def __init__(self, master):
        self.master = master
        master.title("3DOF 로봇팔 시뮬레이터 (클래스 분할)")
        master.geometry("1350x700")

        self.initial_L_values = [3.0, 3.0, 2.0]
        self.initial_q_values_deg = [45.0, -30.0, 30.0]
        self.initial_target_xyphi = [3.0, 3.0, 0.0]

        self.robot = RobotKinematics(*self.initial_L_values)
        self.robot.set_joint_angles_rad(*[np.deg2rad(q) for q in self.initial_q_values_deg])

        self._last_valid_L_values = list(self.initial_L_values)
        self.ik_solution_up = None
        self.ik_solution_down = None
        self._after_id = None

        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self._setup_ui()
        self.update_plot_and_fk()


    def _setup_ui(self):
        main_container = ttk.Frame(self.master)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.left_controls_frame = ttk.Frame(main_container, padding="10", width=380)
        self.left_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        self.left_controls_frame.pack_propagate(False)

        self.ik_controls_frame = ttk.Frame(main_container, padding="10", width=300)
        self.ik_controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.ik_controls_frame.pack_propagate(False)

        self.plot_frame = ttk.Frame(main_container, padding="10")
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Robot Parameters
        params_group = ttk.LabelFrame(self.left_controls_frame, text="로봇 파라미터", padding="10")
        params_group.pack(fill=tk.X, pady=5, anchor=tk.N)

        self.L_vars = []
        self.q_vars = []
        self.L_range_labels = []

        link_names = ["L1", "L2", "L3"]
        for i in range(3):
            self.L_vars.append(tk.DoubleVar(value=round(self.robot.L1 if i==0 else (self.robot.L2 if i==1 else self.robot.L3), DECIMAL_PLACES)))
        self.ensure_initial_link_lengths_valid_gui() # Use current L_vars for initial check

        for i in range(3):
            _, range_label = self._create_param_entry(params_group, f"{link_names[i]} (길이):", self.L_vars[i], LENGTH_STEP, self.parameter_changed_callback, is_length=True, index=i)
            self.L_range_labels.append(range_label)

        angle_names = ["q1 (각도)", "q2 (각도)", "q3 (각도)"]
        current_q_deg = [np.rad2deg(self.robot.q1_rad), np.rad2deg(self.robot.q2_rad), np.rad2deg(self.robot.q3_rad)]
        for i in range(3):
            self.q_vars.append(tk.DoubleVar(value=round(current_q_deg[i], 1)))
            self._create_param_entry(params_group, f"{angle_names[i]}:", self.q_vars[i], ANGLE_STEP, self.parameter_changed_callback, show_range=False)
        
        self.update_length_ranges_display()

        # Current EE Position
        ee_pos_group = ttk.LabelFrame(self.left_controls_frame, text="현재 말단 위치", padding="10")
        ee_pos_group.pack(fill=tk.X, pady=5, anchor=tk.N)
        self.current_ee_x_var = tk.StringVar(value="X: ?")
        self.current_ee_y_var = tk.StringVar(value="Y: ?")
        ttk.Label(ee_pos_group, textvariable=self.current_ee_x_var).pack(anchor=tk.W)
        ttk.Label(ee_pos_group, textvariable=self.current_ee_y_var).pack(anchor=tk.W)

        # Robot Status
        singularity_group = ttk.LabelFrame(self.left_controls_frame, text="로봇 상태", padding="10")
        singularity_group.pack(fill=tk.X, pady=5, anchor=tk.N)
        self.singularity_status_var = tk.StringVar(value="")
        self.singularity_status_label = ttk.Label(singularity_group, textvariable=self.singularity_status_var, foreground="black")
        self.singularity_status_label.pack(anchor=tk.W)

        # Reset Button
        reset_button = ttk.Button(self.left_controls_frame, text="초기화", command=self.reset_simulation)
        reset_button.pack(pady=10, fill=tk.X)

        # IK Controls
        ik_group = ttk.LabelFrame(self.ik_controls_frame, text="역기구학 (목표 위치)", padding="10")
        ik_group.pack(fill=tk.X, pady=5, anchor=tk.N)
        self.target_x_var = tk.DoubleVar(value=self.initial_target_xyphi[0])
        self.target_y_var = tk.DoubleVar(value=self.initial_target_xyphi[1])
        self.phi_e_var = tk.DoubleVar(value=self.initial_target_xyphi[2])
        self._create_param_entry(ik_group, "목표 X:", self.target_x_var, 0.1, None, show_range=False)
        self._create_param_entry(ik_group, "목표 Y:", self.target_y_var, 0.1, None, show_range=False)
        self._create_param_entry(ik_group, "말단 각도 (Phi_e):", self.phi_e_var, 1.0, None, show_range=False)
        ttk.Button(ik_group, text="IK 계산", command=self.calculate_ik_gui).pack(pady=5, fill=tk.X)
        self.ik_sol1_var = tk.StringVar(value="Elbow-up: -")
        self.ik_sol2_var = tk.StringVar(value="Elbow-down: -")
        sol1_button = ttk.Button(ik_group, textvariable=self.ik_sol1_var, command=lambda: self.apply_ik_solution_gui(self.ik_solution_up))
        sol1_button.pack(anchor=tk.W, fill=tk.X)
        sol2_button = ttk.Button(ik_group, textvariable=self.ik_sol2_var, command=lambda: self.apply_ik_solution_gui(self.ik_solution_down))
        sol2_button.pack(anchor=tk.W, fill=tk.X)
        self.ik_status_var = tk.StringVar(value="")
        ttk.Label(ik_group, textvariable=self.ik_status_var, foreground="red").pack(anchor=tk.W)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.line_l1, = self.ax.plot([], [], 'o-', lw=3, color='blue', label='Link 1')
        self.line_l2, = self.ax.plot([], [], 'o-', lw=3, color='green', label='Link 2')
        self.line_l3, = self.ax.plot([], [], 'o-', lw=3, color='red', label='Link 3')
        self.base_dot, = self.ax.plot([0], [0], 'ko', markersize=10, label='Base')
        self.joint1_dot, = self.ax.plot([], [], 'ko', markersize=7)
        self.joint2_dot, = self.ax.plot([], [], 'ko', markersize=7)
        self.joint3_dot, = self.ax.plot([], [], 'ko', markersize=7)
        self.ee_dot, = self.ax.plot([], [], 'ro', markersize=10, label='End-Effector')
        self.target_marker, = self.ax.plot([], [], 'gx', markersize=10, label='Target')
        self.ax.set_xlim(-WORKSPACE_LIMIT -1, WORKSPACE_LIMIT +1)
        self.ax.set_ylim(-WORKSPACE_LIMIT -1, WORKSPACE_LIMIT +1)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True)
        self.ax.set_title("로봇팔 시뮬레이션")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.legend(loc='upper right', fontsize='small')
        workspace_rect = plt.Rectangle((-WORKSPACE_LIMIT, -WORKSPACE_LIMIT),
                                       2*WORKSPACE_LIMIT, 2*WORKSPACE_LIMIT,
                                       fill=False, edgecolor='gray', linestyle='--')
        self.ax.add_patch(workspace_rect)

    def on_closing(self):
        if self._after_id: self.master.after_cancel(self._after_id)
        self.master.quit()
        self.master.destroy()

    def ensure_initial_link_lengths_valid_gui(self):
        # Uses L_vars to set _last_valid_L_values initially
        temp_l_values = []
        for i in range(3):
            try:
                val = round(self.L_vars[i].get(), DECIMAL_PLACES)
            except tk.TclError: # If var is empty or invalid at init
                val = round(self.initial_L_values[i], DECIMAL_PLACES)
            if val < MIN_LINK_LENGTH: val = MIN_LINK_LENGTH
            temp_l_values.append(val)
        
        current_sum = sum(temp_l_values)
        if current_sum > MAX_REACH_SUM:
            excess = current_sum - MAX_REACH_SUM
            reducible_total = sum(max(0, v - MIN_LINK_LENGTH) for v in temp_l_values)
            if reducible_total > 1e-6:
                for i in range(3):
                    reduction = (max(0, temp_l_values[i] - MIN_LINK_LENGTH) / reducible_total) * excess
                    temp_l_values[i] = max(MIN_LINK_LENGTH, temp_l_values[i] - reduction)
        
        for i in range(3):
            rounded_val = round(temp_l_values[i], DECIMAL_PLACES)
            self.L_vars[i].set(rounded_val)
            self._last_valid_L_values[i] = rounded_val
        
        # Sync to robot object
        self.robot.set_link_lengths(*self._last_valid_L_values)


    def _create_param_entry(self, parent, label_text, var, step, command, is_length=False, index=None, show_range=True):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.pack(side=tk.LEFT, padx=5)

        def validate_float(action, val_if_allowed, text_ins, val_type, widget_name):
            if action == '1': # Insert
                if val_if_allowed in ["", "-", ".", "-."]: return True
                try: float(val_if_allowed); return True
                except ValueError: return False
            return True
        vcmd = (parent.register(validate_float), '%d', '%P', '%S', '%v', '%W')
        entry.config(validate='key', validatecommand=vcmd)

        if command:
            entry.bind("<Return>", lambda e, v=var, cmd=command, l=is_length, i=index: self.on_entry_change_gui(v, cmd, l, i))
            entry.bind("<FocusOut>", lambda e, v=var, cmd=command, l=is_length, i=index: self.on_entry_change_gui(v, cmd, l, i))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.LEFT)
        if command:
            up_btn = ttk.Button(btn_frame, text="▲", width=2, command=lambda v=var,s=step,cmd=command,l=is_length,i=index: self.increment_value_gui(v,s,cmd,l,i))
            up_btn.pack(side=tk.TOP, pady=0, ipady=0)
            down_btn = ttk.Button(btn_frame, text="▼", width=2, command=lambda v=var,s=-step,cmd=command,l=is_length,i=index: self.increment_value_gui(v,s,cmd,l,i))
            down_btn.pack(side=tk.BOTTOM, pady=0, ipady=0)
            up_btn.bind("<ButtonPress-1>", lambda e,v=var,s=step,cmd=command,l=is_length,i=index: self.start_continuous_change(v,s,cmd,l,i))
            up_btn.bind("<ButtonRelease-1>", self.stop_continuous_change)
            down_btn.bind("<ButtonPress-1>", lambda e,v=var,s=-step,cmd=command,l=is_length,i=index: self.start_continuous_change(v,s,cmd,l,i))
            down_btn.bind("<ButtonRelease-1>", self.stop_continuous_change)

        range_label_widget = None
        if is_length and show_range:
            range_label_widget = ttk.Label(frame, text="", width=20)
            range_label_widget.pack(side=tk.LEFT, padx=5)
        return entry, range_label_widget

    def parameter_changed_callback(self):
        self.clear_ik_solutions_gui()
        self.update_plot_and_fk_from_ui()

    def on_entry_change_gui(self, var, command_func, is_length_var=False, length_index=None):
        try:
            current_str = var.get()
            if not current_str or current_str in ["-", "."]:
                if is_length_var and length_index is not None:
                    var.set(round(self._last_valid_L_values[length_index], DECIMAL_PLACES))
                return
            
            float_val = float(current_str)
            if is_length_var and length_index is not None:
                L_others_sum = sum(self.L_vars[j].get() for j in range(3) if j != length_index)
                max_this = max(MIN_LINK_LENGTH, MAX_REACH_SUM - L_others_sum)
                clamped = max(MIN_LINK_LENGTH, min(float_val, max_this))
                var.set(round(clamped, DECIMAL_PLACES))
            
            if command_func: command_func()
        except (tk.TclError, ValueError):
            if is_length_var and length_index is not None:
                var.set(round(self._last_valid_L_values[length_index], DECIMAL_PLACES))
            if command_func: command_func()


    def increment_value_gui(self, var, step, command_func, is_length_var=False, length_index=None):
        try: current_val = var.get()
        except tk.TclError: current_val = self._last_valid_L_values[length_index] if is_length_var and length_index is not None else 0.0
        
        new_val_attempt = round(current_val + step, DECIMAL_PLACES + 1)
        if is_length_var and length_index is not None:
            L_others_sum = sum(self.L_vars[j].get() for j in range(3) if j != length_index)
            max_this = max(MIN_LINK_LENGTH, MAX_REACH_SUM - L_others_sum)
            clamped = max(MIN_LINK_LENGTH, min(new_val_attempt, max_this))
            var.set(round(clamped, DECIMAL_PLACES))
        else:
            var.set(round(new_val_attempt, 1))
        
        if command_func: command_func()

    def start_continuous_change(self, var, step, command, is_length, index):
        self.stop_continuous_change()
        self.increment_value_gui(var, step, command, is_length, index)
        self._after_id = self.master.after(100, lambda: self.start_continuous_change(var,step,command,is_length,index))

    def stop_continuous_change(self, event=None):
        if self._after_id: self.master.after_cancel(self._after_id); self._after_id = None

    def _sync_robot_params_from_gui(self):
        # Sync link lengths from L_vars to robot object and _last_valid_L_values
        for i in range(3):
            try:
                val = round(self.L_vars[i].get(), DECIMAL_PLACES)
                self._last_valid_L_values[i] = val # Update internal cache
                # self.L_vars[i].set(val) # Ensure GUI var is also rounded
            except tk.TclError: # If GUI var is invalid, use internal cache
                val = self._last_valid_L_values[i]
                self.L_vars[i].set(val)
        self.robot.set_link_lengths(*self._last_valid_L_values)

        # Sync joint angles from q_vars to robot object
        q_rad_gui = [np.deg2rad(self.q_vars[i].get()) for i in range(3)]
        self.robot.set_joint_angles_rad(*q_rad_gui)


    def update_length_ranges_display(self):
        self._sync_robot_params_from_gui() # Ensure robot L values are current
        current_L_values = [self.robot.L1, self.robot.L2, self.robot.L3]

        for i in range(3):
            other_L_sum = sum(current_L_values[j] for j in range(3) if j != i)
            max_li = round(max(MIN_LINK_LENGTH, MAX_REACH_SUM - other_L_sum), DECIMAL_PLACES)
            min_li = round(MIN_LINK_LENGTH, DECIMAL_PLACES)
            
            if self.L_range_labels[i]:
                self.L_range_labels[i].config(text=f"[{min_li:.{DECIMAL_PLACES}f} ~ {max_li:.{DECIMAL_PLACES}f}]")
                current_li_val_gui = round(self.L_vars[i].get(), DECIMAL_PLACES) # Value in GUI
                if not (min_li - 1e-6 <= current_li_val_gui <= max_li + 1e-6):
                    self.L_range_labels[i].config(foreground="red")
                    # self.L_entries[i].config(foreground="red") # Entry text color change can be distracting
                else:
                    self.L_range_labels[i].config(foreground="black")
                    # self.L_entries[i].config(foreground="black")

    def update_plot_and_fk_from_ui(self):
        self._sync_robot_params_from_gui() # Sync L and q from GUI to robot object
        self.update_length_ranges_display() # Update ranges based on synced L
        self.update_plot_and_fk()           # Perform FK and plot with robot object's state

    def update_robot_status_display(self):
        is_singular, messages = self.robot.check_current_angles_singularity()
        if is_singular:
            self.singularity_status_var.set("충돌 감지: " + ", ".join(messages))
            self.singularity_status_label.config(foreground="red")
        else:
            self.singularity_status_var.set("상태: 양호")
            self.singularity_status_label.config(foreground="black")

    def update_plot_and_fk(self):
        # Assumes robot object (self.robot) has the correct L and q values
        P0, P1, P2, P3 = self.robot.forward_kinematics()
        self.line_l1.set_data([P0[0], P1[0]], [P0[1], P1[1]])
        self.line_l2.set_data([P1[0], P2[0]], [P1[1], P2[1]])
        self.line_l3.set_data([P2[0], P3[0]], [P2[1], P3[1]])
        self.joint1_dot.set_data([P0[0]],[P0[1]]); self.joint2_dot.set_data([P1[0]],[P1[1]])
        self.joint3_dot.set_data([P2[0]],[P2[1]]); self.ee_dot.set_data([P3[0]],[P3[1]])
        self.current_ee_x_var.set(f"X: {P3[0]:.3f}")
        self.current_ee_y_var.set(f"Y: {P3[1]:.3f}")
        self.ee_dot.set_markerfacecolor('magenta' if abs(P3[0]) > WORKSPACE_LIMIT or abs(P3[1]) > WORKSPACE_LIMIT else 'red')
        
        self.update_robot_status_display() # Update singularity status based on current robot angles
        self.canvas.draw_idle()

    def clear_ik_solutions_gui(self):
        self.ik_solution_up = None
        self.ik_solution_down = None
        self.ik_sol1_var.set("Elbow-up: -")
        self.ik_sol2_var.set("Elbow-down: -")
        # self.ik_status_var.set("")

    def calculate_ik_gui(self):
        self._sync_robot_params_from_gui() # Ensure robot object has current link lengths from GUI
        
        target_x = self.target_x_var.get()
        target_y = self.target_y_var.get()
        phi_e_rad = np.deg2rad(self.phi_e_var.get())

        self.target_marker.set_data([target_x], [target_y])
        self.clear_ik_solutions_gui()

        # Use robot object's current link lengths for IK calculation
        current_L_sum = self.robot.L1 + self.robot.L2 + self.robot.L3
        if np.sqrt(target_x**2 + target_y**2) > current_L_sum + 1e-6:
            self.ik_status_var.set("이동 불가 (목표가 너무 멉니다)")
            self.canvas.draw_idle(); return
        
        # Further reachability for L1,L2 to wrist
        xw = target_x - self.robot.L3 * np.cos(phi_e_rad)
        yw = target_y - self.robot.L3 * np.sin(phi_e_rad)
        if np.sqrt(xw**2+yw**2) < np.abs(self.robot.L1-self.robot.L2)-1e-6 or \
           np.sqrt(xw**2+yw**2) > self.robot.L1+self.robot.L2+1e-6 :
            self.ik_status_var.set("이동 불가 (L1,L2로 손목 도달 불가)")
            self.canvas.draw_idle(); return

        temp_sol_up, temp_sol_down = self.robot.inverse_kinematics(target_x, target_y, phi_e_rad)
        
        up_singular = self.robot.is_solution_near_singularity(temp_sol_up)
        down_singular = self.robot.is_solution_near_singularity(temp_sol_down)

        self.ik_solution_up = temp_sol_up if not up_singular else None
        self.ik_solution_down = temp_sol_down if not down_singular else None
        
        valid_solutions_exist = False
        if self.ik_solution_up:
            q1,q2,q3 = np.rad2deg(self.ik_solution_up)
            self.ik_sol1_var.set(f"Elbow-up: ({q1:.1f},{q2:.1f},{q3:.1f})°")
            valid_solutions_exist = True
        elif temp_sol_up: self.ik_sol1_var.set("Elbow-up: (특이점)")
        else: self.ik_sol1_var.set("Elbow-up: (해 없음)")

        if self.ik_solution_down:
            q1,q2,q3 = np.rad2deg(self.ik_solution_down)
            self.ik_sol2_var.set(f"Elbow-down: ({q1:.1f},{q2:.1f},{q3:.1f})°")
            valid_solutions_exist = True
        elif temp_sol_down: self.ik_sol2_var.set("Elbow-down: (특이점)")
        else: self.ik_sol2_var.set("Elbow-down: (해 없음)")

        if valid_solutions_exist:
            self.ik_status_var.set("IK 해법 적용 가능.")
            if self.ik_solution_up: self.apply_ik_solution_gui(self.ik_solution_up)
            elif self.ik_solution_down: self.apply_ik_solution_gui(self.ik_solution_down)
        else:
            self.ik_status_var.set("이동 불가 (유효 해법 없음 또는 특이점)")
        self.canvas.draw_idle()

    def apply_ik_solution_gui(self, solution_rad):
        if solution_rad:
            q1_r, q2_r, q3_r = solution_rad
            self.q_vars[0].set(round(np.rad2deg(q1_r),1))
            self.q_vars[1].set(round(np.rad2deg(q2_r),1))
            self.q_vars[2].set(round(np.rad2deg(q3_r),1))
            
            # This will trigger parameter_changed_callback through q_vars trace,
            # which calls clear_ik_solutions_gui and update_plot_and_fk_from_ui.
            # However, we want to keep the IK solutions visible after applying one.
            # So, call update_plot_and_fk_from_ui directly but don't clear solutions here.
            self.update_plot_and_fk_from_ui()
            # Re-affirm the IK solution text as parameter_changed_callback might clear it.
            if solution_rad == self.ik_solution_up and self.ik_solution_up:
                 q1u, q2u, q3u = np.rad2deg(self.ik_solution_up)
                 self.ik_sol1_var.set(f"Elbow-up: ({q1u:.1f}, {q2u:.1f}, {q3u:.1f})°")
            if solution_rad == self.ik_solution_down and self.ik_solution_down:
                 q1d, q2d, q3d = np.rad2deg(self.ik_solution_down)
                 self.ik_sol2_var.set(f"Elbow-down: ({q1d:.1f}, {q2d:.1f}, {q3d:.1f})°")

        else:
            self.ik_status_var.set("적용할 유효한 IK 해법이 없습니다.")
            # Make sure button text reflects no solution if clicked when solution is None
            if solution_rad is self.ik_solution_up: self.ik_sol1_var.set("Elbow-up: -")
            if solution_rad is self.ik_solution_down: self.ik_sol2_var.set("Elbow-down: -")


    def reset_simulation(self):
        for i in range(3):
            self.L_vars[i].set(round(self.initial_L_values[i], DECIMAL_PLACES))
        self._last_valid_L_values = list(self.initial_L_values)
        self.ensure_initial_link_lengths_valid_gui() # Syncs L_vars, _last_valid_L_values, and robot.L

        for i in range(3):
            self.q_vars[i].set(self.initial_q_values_deg[i])
        # Sync q_vars to robot object after reset
        q_rad_initial = [np.deg2rad(q_deg) for q_deg in self.initial_q_values_deg]
        self.robot.set_joint_angles_rad(*q_rad_initial)
            
        self.target_x_var.set(self.initial_target_xyphi[0])
        self.target_y_var.set(self.initial_target_xyphi[1])
        self.phi_e_var.set(self.initial_target_xyphi[2])
        self.target_marker.set_data([], []) 

        self.clear_ik_solutions_gui()
        self.ik_status_var.set("")
        self.update_plot_and_fk_from_ui()


if __name__ == '__main__':
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        # Attempt to use a more modern theme if available
        themes = style.theme_names()
        if 'clam' in themes: style.theme_use('clam')
        elif 'alt' in themes: style.theme_use('alt')
        elif 'vista' in themes: style.theme_use('vista') # Windows
        elif 'aqua' in themes: style.theme_use('aqua') # macOS
    except tk.TclError:
        pass # Default theme will be used

    app = RobotArmGUI(root)
    root.mainloop()
