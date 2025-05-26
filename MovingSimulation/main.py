"""
Robot Kinematics Simulation Program - 핵심 문제 해결 버전
메인 실행 파일 - MovingSimulation/main.py (4개 핵심 문제 해결)

해결된 문제들:
1. ✅ 상태창 텍스트 줄간격 문제 완전 해결
2. ✅ Auto Update 코드 완전 제거 (더 직관적인 Calculate 버튼으로 대체)
3. ✅ 목표점 위치와 시뮬레이션 정확성 추적 시스템 구축
4. ✅ 시뮬레이션 초기화 및 시각적 동작 일관성 보장
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import yaml
import csv
from datetime import datetime

# 사용자 정의 모듈 import
from robot_kinematics import RobotKinematics
from dh_parameters import DHParameterManager
from trajectory_planner import TrajectoryPlanner
from visualization import RobotVisualizer
from utils import Utils

class RobotSimulationGUI:
    def __init__(self, root):
        """로봇 시뮬레이션 GUI 초기화 - 핵심 문제 해결 버전"""
        self.root = root
        self.root.title("Robot Kinematics Simulation - 핵심 문제 해결 버전")
        self.root.geometry("1600x1000")
        
        # 핵심 클래스 초기화
        self.robot_kinematics = RobotKinematics()
        self.dh_manager = DHParameterManager()
        self.trajectory_planner = TrajectoryPlanner()
        self.visualizer = RobotVisualizer()
        self.utils = Utils()
        
        # 시뮬레이션 상태 변수들
        self.current_dof = 1
        self.robot_type = "Forward"
        self.joint_angles = [0.0]
        self.target_position = [30.0, 0.0, 15.0]
        self.target_orientation = [0.0, 0.0, 0.0]
        self.dh_params = []
        self.simulation_running = False
        
        # 문제 3 해결: 정확성 추적을 위한 정밀한 허용 오차 설정
        self.position_tolerance_mm = 5.0  # 5mm 허용 오차
        self.angle_tolerance_deg = 0.5    # 0.5도 허용 오차
        
        # 버튼 누름 상태 관리
        self.button_pressed = {}
        self.button_press_count = {}
        
        # Inverse Kinematics 해 관리
        self.ik_solutions = []
        self.selected_ik_solution = 0
        
        # 경로 및 분석 데이터
        self.current_path = None
        self.trajectory_history = []
        
        # 문제 4 해결: 시각적 상태 관리
        self.simulation_state = {
            'is_first_run': True,
            'last_target_position': None,
            'last_joint_angles': None,
            'animation_step': 0
        }
        
        # GUI 요소 관리
        self.dh_frame = None
        self.dh_entries = {}
        self.dh_buttons = {}
        self.ik_solutions_frame = None
        self.joint_frame = None
        self.target_frame = None
        self.input_entries = {}
        self.input_buttons = {}
        self.result_text = None
        
        # 컨트롤 위젯들
        self.control_widgets = []
        
        # 단위 변환 상수
        self.CM_TO_M = 0.01
        self.M_TO_CM = 100.0
        
        # GUI 초기화
        self.setup_gui()
        self.setup_robot_visualization()
        self.load_initial_configuration()
        
    def setup_gui(self):
        self.root.geometry("1700x1000")  # 기존 1600에서 1700으로 증가
        
        # 메인 PanedWindow - 좌우 분할
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 좌측 컨트롤 패널
        control_frame = ttk.Frame(main_paned)
        main_paned.add(control_frame)
        
        # 우측 시각화 패널
        viz_frame = ttk.Frame(main_paned)
        main_paned.add(viz_frame)
        
        # 각 패널 설정
        self.setup_control_panel(control_frame)
        self.setup_visualization_panel(viz_frame)
        
        # 핵심: GUI가 완전히 로드된 후 분할 위치를 강제로 설정
        self.root.after(100, self.set_initial_panel_sizes)

    def set_initial_panel_sizes(self):
        """초기 패널 크기를 강제로 설정"""
        try:
            # PanedWindow 찾기
            for child in self.root.winfo_children():
                if isinstance(child, ttk.PanedWindow):
                    # 첫 번째 패널(컨트롤 패널)을 500픽셀로 고정
                    child.sashpos(0, 500)  # 0번째 구분선을 500픽셀 위치에 설정
                    break
        except:
            pass  # 오류가 발생해도 프로그램은 계속 실행
    def setup_control_panel(self, parent):
        """컨트롤 패널 설정"""
        # 로봇 구성 섹션
        type_frame = ttk.LabelFrame(parent, text="Robot Configuration")
        type_frame.pack(fill=tk.X, pady=5)
        self.control_widgets.append(type_frame)
        
        # 로봇 타입 선택
        ttk.Label(type_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.type_var = tk.StringVar(value="Forward")
        type_combo = ttk.Combobox(type_frame, textvariable=self.type_var, 
                                 values=["Forward", "Inverse"], state="readonly", width=15)
        type_combo.grid(row=0, column=1, padx=5, pady=2)
        type_combo.bind('<<ComboboxSelected>>', self.on_robot_type_change)
        
        # DOF 선택
        ttk.Label(type_frame, text="DOF:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.dof_var = tk.IntVar(value=1)
        dof_combo = ttk.Combobox(type_frame, textvariable=self.dof_var,
                                values=[1, 2, 3, 4, 5, 6], state="readonly", width=15)
        dof_combo.grid(row=1, column=1, padx=5, pady=2)
        dof_combo.bind('<<ComboboxSelected>>', self.on_dof_change)
        
        # 문제 3 해결: 정확한 End-Effector 위치 표시 및 목표점과의 오차 분석
        self.current_ee_frame = ttk.Frame(type_frame)
        self.current_ee_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        self.current_ee_label = ttk.Label(self.current_ee_frame, 
                                         text="Current EE: X=30.0, Y=0.0, Z=15.0 cm", 
                                         font=('Arial', 9), foreground='darkgreen')
        self.current_ee_label.pack()
        
        # 문제 3 해결: 목표점과의 오차 표시 라벨 추가
        self.position_error_label = ttk.Label(self.current_ee_frame, 
                                            text="Target Error: 0.0 mm", 
                                            font=('Arial', 8), foreground='darkblue')
        self.position_error_label.pack()
        
        # DH 파라미터 섹션
        dh_frame = ttk.LabelFrame(parent, text="DH Parameters (cm)")
        dh_frame.pack(fill=tk.X, pady=5)
        self.control_widgets.append(dh_frame)
        self.setup_dh_parameter_inputs(dh_frame)
        
        # IK Solutions 프레임
        self.setup_ik_solutions_frame(parent)
        
        # 현재 관절 각도 표시
        current_joints_frame = ttk.LabelFrame(parent, text="Current Joint Angles (deg)")
        current_joints_frame.pack(fill=tk.X, pady=5)
        self.control_widgets.append(current_joints_frame)
        self.setup_current_joint_display(current_joints_frame)
        
        # 목표 위치 입력
        target_frame = ttk.LabelFrame(parent, text="Target Position & Orientation")
        target_frame.pack(fill=tk.X, pady=5)
        self.control_widgets.append(target_frame)
        self.setup_target_inputs(target_frame)
        
        # 문제 1 해결: 상태창 줄간격 문제 해결된 결과 표시 섹션
        result_frame = ttk.LabelFrame(parent, text="Analysis Results")
        result_frame.pack(fill=tk.X, pady=5)
        
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text = tk.Text(text_frame, height=8, width=50, font=('Courier', 9))
        result_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 제어 버튼들
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=5)
        self.control_widgets.append(button_frame)
        self.setup_control_buttons(button_frame)
        
    def setup_ik_solutions_frame(self, parent):
        """IK Solutions 프레임 설정"""
        self.ik_solutions_frame = ttk.LabelFrame(parent, text="IK Solutions")
        self.ik_solutions_frame.pack(fill=tk.X, pady=5)
        
        # 스크롤 가능한 캔버스 생성
        canvas_frame = ttk.Frame(self.ik_solutions_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ik_solutions_canvas = tk.Canvas(canvas_frame, height=120)
        scrollbar_ik = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.ik_solutions_canvas.yview)
        self.ik_solutions_scrollable_frame = ttk.Frame(self.ik_solutions_canvas)
        
        self.ik_solutions_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.ik_solutions_canvas.configure(scrollregion=self.ik_solutions_canvas.bbox("all"))
        )
        
        self.ik_solutions_canvas.create_window((0, 0), window=self.ik_solutions_scrollable_frame, anchor="nw")
        self.ik_solutions_canvas.configure(yscrollcommand=scrollbar_ik.set)
        
        # 마우스 휠 스크롤 지원
        def _on_mousewheel_ik(event):
            self.ik_solutions_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.ik_solutions_canvas.bind_all("<MouseWheel>", _on_mousewheel_ik)
        
        self.ik_solutions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_ik.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 처음에는 숨김
        self.ik_solutions_frame.pack_forget()
        
    def setup_dh_parameter_inputs(self, parent):
        """DH 파라미터 입력 필드 설정"""
        self.dh_frame = ttk.Frame(parent)
        self.dh_frame.pack(fill=tk.X, pady=5)
        
        self.create_dh_headers()
        self.update_dh_parameter_inputs()
        
        # YAML 파일 로드/저장 버튼들
        yaml_frame = ttk.Frame(parent)
        yaml_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(yaml_frame, text="Load YAML", 
                  command=self.load_yaml_parameters).pack(side=tk.LEFT, padx=2)
        ttk.Button(yaml_frame, text="Save YAML", 
                  command=self.save_yaml_parameters).pack(side=tk.LEFT, padx=2)
        ttk.Button(yaml_frame, text="Load Preset", 
                  command=self.load_preset_robot).pack(side=tk.LEFT, padx=2)
    
    def create_dh_headers(self):
        """DH 파라미터 테이블 헤더 생성"""
        if hasattr(self, 'header_frame') and self.header_frame is not None:
            self.header_frame.destroy()
        
        self.header_frame = ttk.Frame(self.dh_frame)
        self.header_frame.pack(fill=tk.X, pady=2)
        
        headers = ["Link", "a (cm)", "", "", "α (deg)", "", "", "d (cm)", "", "", "θ (deg)", "", ""]
        widths = [6, 8, 3, 3, 8, 3, 3, 8, 3, 3, 8, 3, 3]
        
        for i, (header, width) in enumerate(zip(headers, widths)):
            if header:
                label = ttk.Label(self.header_frame, text=header, font=('Arial', 9, 'bold'), width=width)
                label.grid(row=0, column=i, padx=1, pady=2)
    
    def update_dh_parameter_inputs(self):
        """DH 파라미터 입력 필드 업데이트"""
        if hasattr(self, 'dh_input_frame') and self.dh_input_frame is not None:
            self.dh_input_frame.destroy()
        
        self.dh_input_frame = ttk.Frame(self.dh_frame)
        self.dh_input_frame.pack(fill=tk.X, pady=2)
        
        self.dh_entries = {}
        self.dh_buttons = {}
        
        if len(self.dh_params) != self.current_dof:
            self.load_default_dh_params_for_dof(self.current_dof)
        
        for i in range(self.current_dof):
            self.dh_entries[i] = {}
            self.dh_buttons[i] = {}
            
            row = i
            col = 0
            
            # 링크 번호 라벨
            link_label = ttk.Label(self.dh_input_frame, text=f"Link {i+1}", width=6)
            link_label.grid(row=row, column=col, padx=1, pady=1)
            col += 1
            
            param_names = ['a', 'alpha', 'd', 'theta']
            param_steps = [0.5, 1.0, 0.5, 1.0]
            
            for j, (param_name, step) in enumerate(zip(param_names, param_steps)):
                # 감소 버튼
                dec_btn = ttk.Button(self.dh_input_frame, text="◀", width=3)
                dec_btn.grid(row=row, column=col, padx=1, pady=1)
                col += 1
                
                # 입력 필드
                entry = ttk.Entry(self.dh_input_frame, width=8, justify='center')
                entry.grid(row=row, column=col, padx=1, pady=1)
                
                if i < len(self.dh_params) and j < len(self.dh_params[i]):
                    entry.insert(0, f"{self.dh_params[i][j]:.1f}")
                else:
                    entry.insert(0, "0.0")
                
                entry.bind('<KeyRelease>', self.on_parameter_change)
                entry.bind('<FocusOut>', self.on_parameter_change)
                
                col += 1
                
                # 증가 버튼
                inc_btn = ttk.Button(self.dh_input_frame, text="▶", width=3)
                inc_btn.grid(row=row, column=col, padx=1, pady=1)
                col += 1
                
                # 버튼 이벤트 바인딩
                dec_btn.bind('<Button-1>', 
                           lambda e, idx=i, param=param_name, s=-step: self.start_dh_button_press(idx, param, s))
                dec_btn.bind('<ButtonRelease-1>', 
                           lambda e, idx=i, param=param_name: self.stop_dh_button_press(idx, param))
                
                inc_btn.bind('<Button-1>', 
                           lambda e, idx=i, param=param_name, s=step: self.start_dh_button_press(idx, param, s))
                inc_btn.bind('<ButtonRelease-1>', 
                           lambda e, idx=i, param=param_name: self.stop_dh_button_press(idx, param))
                
                self.dh_entries[i][param_name] = entry
                self.dh_buttons[i][param_name] = {'dec': dec_btn, 'inc': inc_btn}

    def start_dh_button_press(self, link_idx, param_name, step):
        """DH 파라미터 버튼 연속 누름 시작"""
        button_key = f"dh_{link_idx}_{param_name}"
        self.button_pressed[button_key] = True
        self.button_press_count[button_key] = 0
        
        self.change_dh_parameter_value(link_idx, param_name, step)
        self.root.after(300, lambda: self.continuous_dh_button_press(link_idx, param_name, step))
    
    def stop_dh_button_press(self, link_idx, param_name):
        """DH 파라미터 버튼 연속 누름 중지"""
        button_key = f"dh_{link_idx}_{param_name}"
        self.button_pressed[button_key] = False
        self.button_press_count[button_key] = 0
    
    def continuous_dh_button_press(self, link_idx, param_name, step):
        """DH 파라미터 버튼 연속 누름 처리"""
        button_key = f"dh_{link_idx}_{param_name}"
        
        if self.button_pressed.get(button_key, False):
            self.button_press_count[button_key] += 1
            
            count = self.button_press_count[button_key]
            if count < 5:
                current_step = step
            elif count < 10:
                current_step = step * 2
            else:
                current_step = step * 3
            
            self.change_dh_parameter_value(link_idx, param_name, current_step)
            
            interval = max(50, 150 - count * 10)
            self.root.after(interval, lambda: self.continuous_dh_button_press(link_idx, param_name, step))
    
    def change_dh_parameter_value(self, link_idx, param_name, step):
        """DH 파라미터 값 변경"""
        try:
            current_value = float(self.dh_entries[link_idx][param_name].get())
            new_value = current_value + step
            
            if param_name in ['a', 'd']:
                new_value = max(-200, min(200, new_value))
            elif param_name in ['alpha', 'theta']:
                new_value = max(-360, min(360, new_value))
            
            self.dh_entries[link_idx][param_name].delete(0, tk.END)
            self.dh_entries[link_idx][param_name].insert(0, f"{new_value:.1f}")
            
            self.on_parameter_change()
            
        except ValueError:
            pass

    def setup_current_joint_display(self, parent):
        """현재 관절 각도 표시/조작 패널"""
        self.joint_frame = ttk.Frame(parent)
        self.joint_frame.pack(fill=tk.X, pady=5)
        self.update_joint_display()
    
    def update_joint_display(self):
        """관절 각도 표시 업데이트"""
        if hasattr(self, 'joint_input_frame') and self.joint_input_frame is not None:
            self.joint_input_frame.destroy()
        
        self.joint_input_frame = ttk.Frame(self.joint_frame)
        self.joint_input_frame.pack(fill=tk.X, pady=2)
        
        self.input_entries = {}
        self.input_buttons = {}
        
        for i in range(self.current_dof):
            joint_frame = ttk.Frame(self.joint_input_frame)
            joint_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(joint_frame, text=f"Joint {i+1}:", width=8).pack(side=tk.LEFT)
            
            if self.robot_type == "Forward":
                # Forward 모드: 조작 가능
                dec_btn = ttk.Button(joint_frame, text="◀", width=3)
                dec_btn.pack(side=tk.LEFT, padx=2)
                dec_btn.bind('<Button-1>', lambda e, idx=i: self.start_button_press(idx, -1))
                dec_btn.bind('<ButtonRelease-1>', lambda e, idx=i: self.stop_button_press(idx))
                
                entry = ttk.Entry(joint_frame, width=10, justify='center')
                entry.pack(side=tk.LEFT, padx=2)
                if i < len(self.joint_angles):
                    entry.insert(0, f"{self.joint_angles[i]:.2f}")
                else:
                    entry.insert(0, "0.00")
                
                entry.bind('<KeyRelease>', self.on_joint_angle_change)
                entry.bind('<FocusOut>', self.on_joint_angle_change)
                
                inc_btn = ttk.Button(joint_frame, text="▶", width=3)
                inc_btn.pack(side=tk.LEFT, padx=2)
                inc_btn.bind('<Button-1>', lambda e, idx=i: self.start_button_press(idx, 1))
                inc_btn.bind('<ButtonRelease-1>', lambda e, idx=i: self.stop_button_press(idx))
                
                self.input_entries[i] = entry
                self.input_buttons[i] = {'dec': dec_btn, 'inc': inc_btn}
            else:
                # Inverse 모드: 표시만
                angle_label = ttk.Label(joint_frame, text=f"{self.joint_angles[i]:.2f}°", 
                                      width=12, relief='sunken', anchor='center')
                angle_label.pack(side=tk.LEFT, padx=2)
    
    def setup_target_inputs(self, parent):
        """목표 위치 및 자세 입력 설정"""
        self.target_frame = ttk.Frame(parent)
        self.target_frame.pack(fill=tk.X, pady=5)
        
        # 목표 위치 입력
        pos_frame = ttk.LabelFrame(self.target_frame, text="Target Position (cm)")
        pos_frame.pack(fill=tk.X, pady=2)
        
        pos_labels = ["X:", "Y:", "Z:"]
        self.target_pos_entries = {}
        
        for i, label in enumerate(pos_labels):
            pos_row = ttk.Frame(pos_frame)
            pos_row.pack(fill=tk.X, pady=1)
            
            ttk.Label(pos_row, text=label, width=3).pack(side=tk.LEFT)
            entry = ttk.Entry(pos_row, width=12, justify='center')
            entry.pack(side=tk.LEFT, padx=5)
            
            if i < len(self.target_position):
                entry.insert(0, f"{self.target_position[i]:.2f}")
            else:
                entry.insert(0, "0.00")
            
            entry.bind('<KeyRelease>', self.on_target_change)
            entry.bind('<FocusOut>', self.on_target_change)
            
            self.target_pos_entries[i] = entry
        
        # 목표 자세 입력 (3DOF 이상일 때만 표시)
        self.ori_frame = ttk.LabelFrame(self.target_frame, text="Target Orientation (deg) - 3DOF+ Only")
        
        ori_labels = ["Roll:", "Pitch:", "Yaw:"]
        self.target_ori_entries = {}
        
        for i, label in enumerate(ori_labels):
            ori_row = ttk.Frame(self.ori_frame)
            ori_row.pack(fill=tk.X, pady=1)
            
            ttk.Label(ori_row, text=label, width=6).pack(side=tk.LEFT)
            entry = ttk.Entry(ori_row, width=12, justify='center')
            entry.pack(side=tk.LEFT, padx=5)
            
            if i < len(self.target_orientation):
                entry.insert(0, f"{self.target_orientation[i]:.2f}")
            else:
                entry.insert(0, "0.00")
            
            entry.bind('<KeyRelease>', self.on_target_change)
            entry.bind('<FocusOut>', self.on_target_change)
            
            self.target_ori_entries[i] = entry

        preview_frame = ttk.Frame(self.target_frame)
        preview_frame.pack(fill=tk.X, pady=3)
        self.preview_target_btn = ttk.Button(preview_frame, 
                                            text="👁️ Preview Target", 
                                            command=self.preview_target_position,
                                            width=15)
        self.preview_target_btn.pack(side=tk.LEFT, padx=5)
        self.preview_status_label = ttk.Label(preview_frame, 
                                            text="",
                                            font=('Arial', 8), foreground='darkblue')
        self.preview_status_label.pack(side=tk.LEFT, padx=10)
        
        self.update_target_inputs_visibility()
    
    def update_target_inputs_visibility(self):
        """DOF에 따른 목표 입력 가시성 업데이트"""
        if self.current_dof >= 3:
            self.ori_frame.pack(fill=tk.X, pady=2)
        else:
            self.ori_frame.pack_forget()
    
    def setup_control_buttons(self, parent):
        """제어 버튼 설정 - 문제 2 해결: Auto Update 완전 제거"""
        # 첫 번째 줄: 주요 기능
        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, pady=2)
    
        self.calculate_btn = ttk.Button(row1, text="🔄 Calculate", 
                                    command=self.calculate_inverse_kinematics, width=12)
            
        # 핵심 시뮬레이션 버튼
        self.simulation_btn = ttk.Button(row1, text="🎯 Run Simulation", 
                                        command=self.run_goal_oriented_simulation, width=15,
                                        style='Accent.TButton')
        self.simulation_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(row1, text="Reset", 
                  command=self.reset_to_default, width=8).pack(side=tk.LEFT, padx=2)
        
        # 두 번째 줄: 보조 기능
        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(row2, text="Random Params", 
                  command=self.generate_random_parameters, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Analyze Workspace", 
                  command=self.analyze_workspace, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="Save Results", 
                  command=self.save_results, width=12).pack(side=tk.LEFT, padx=2)
        
        # 시뮬레이션 설명
        info_label = ttk.Label(parent, 
                              text="🎯 Run Simulation: 현재 위치에서 목표 위치까지의 최적 경로로 이동", 
                              font=('Arial', 9), foreground='darkblue')
        info_label.pack(pady=5)
    
    def setup_visualization_panel(self, parent):
        """시각화 패널 설정"""
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_robot_visualization(self):
        """로봇 시각화 초기 설정"""
        self.ax.clear()
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(0.0, 1.5)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Robot Kinematics Simulation')
        self.canvas.draw()
    
    # ========== 문제 해결을 위한 핵심 메서드들 ==========
    
    def add_status_message(self, message):
        """문제 1 해결: 상태창 줄간격 문제 완전 해결"""
        if self.result_text is not None:
            # 현재 텍스트의 마지막이 비어있는지 확인
            current_content = self.result_text.get("1.0", tk.END)
            
            # 빈 줄 제거 로직: 연속된 줄바꿈을 하나로 통합
            if current_content.strip() and not current_content.endswith('\n'):
                self.result_text.insert(tk.END, '\n')
            
            # 메시지 추가 (줄바꿈 없이)
            self.result_text.insert(tk.END, message)
            
            # 마지막에 단일 줄바꿈만 추가
            if not message.endswith('\n'):
                self.result_text.insert(tk.END, '\n')
            
            # 스크롤을 맨 아래로
            self.result_text.see(tk.END)
        
    def calculate_target_position_error(self, current_ee_pos_cm, target_pos_cm):
        """문제 3 해결: 목표점과 현재 위치의 정확한 오차 계산"""
        error_vector = np.array(target_pos_cm) - np.array(current_ee_pos_cm)
        error_distance_mm = np.linalg.norm(error_vector) * 10  # cm를 mm로 변환
        
        # 개별 축 오차도 계산
        x_error_mm = abs(error_vector[0]) * 10
        y_error_mm = abs(error_vector[1]) * 10
        z_error_mm = abs(error_vector[2]) * 10
        
        return {
            'total_error_mm': error_distance_mm,
            'x_error_mm': x_error_mm,
            'y_error_mm': y_error_mm,
            'z_error_mm': z_error_mm,
            'is_within_tolerance': error_distance_mm <= self.position_tolerance_mm
        }
    
    def reset_simulation_state(self):
        """문제 4 해결: 시뮬레이션 상태 완전 초기화"""
        self.simulation_state = {
            'is_first_run': True,
            'last_target_position': None,
            'last_joint_angles': None,
            'animation_step': 0
        }
        self.trajectory_history = []
        self.current_path = None
        
        # 시각적 상태도 초기화
        self.ax.clear()
        self.setup_robot_visualization()
        self.update_robot_display()
    
    # ========== 핵심 이벤트 핸들러들 ==========
    
    def on_parameter_change(self, event=None):
        """DH 파라미터 변경시 실시간 업데이트"""
        self.root.after(200, self.update_robot_display)
    
    def on_joint_angle_change(self, event=None):
        """관절 각도 변경시 처리"""
        if self.robot_type == "Forward":
            self.root.after(100, self.update_robot_display)
    
    def on_target_change(self, event=None):
        """목표 위치 변경시 처리"""
        try:
            for i in range(3):
                if i in self.target_pos_entries:
                    self.target_position[i] = float(self.target_pos_entries[i].get())
        except ValueError:
            pass
        
        try:
            for i in range(3):
                if i in self.target_ori_entries:
                    self.target_orientation[i] = float(self.target_ori_entries[i].get())
        except ValueError:
            pass
        
        # 문제 4 해결: 목표 위치 변경시 시뮬레이션 상태 초기화
        if self.simulation_state['last_target_position'] != self.target_position:
            self.simulation_state['is_first_run'] = True
            self.simulation_state['last_target_position'] = self.target_position.copy()
    
    def preview_target_position(self):
        """타겟 위치 미리보기"""
        try:
            target_pos_cm = []
            for i in range(3):
                try:
                    pos = float(self.target_pos_entries[i].get())
                    target_pos_cm.append(pos)
                except (ValueError, KeyError):
                    target_pos_cm.append(0.0)
            
            target_ori = []
            for i in range(3):
                try:
                    ori = float(self.target_ori_entries[i].get())
                    target_ori.append(ori)
                except (ValueError, KeyError):
                    target_ori.append(0.0)
            
            old_target_pos = self.target_position.copy()
            old_target_ori = self.target_orientation.copy()
            
            self.target_position = target_pos_cm
            self.target_orientation = target_ori
            
            self.update_robot_display()
            
            if hasattr(self, 'preview_status_label'):
                self.preview_status_label.config(
                    text=f"Preview: X={target_pos_cm[0]:.1f}, Y={target_pos_cm[1]:.1f}, Z={target_pos_cm[2]:.1f}"
                )
            
            self.add_status_message(f"👁️ 타겟 미리보기: X={target_pos_cm[0]:.1f}, Y={target_pos_cm[1]:.1f}, Z={target_pos_cm[2]:.1f}")
            if self.current_dof >= 3:
                self.add_status_message(f"   자세: Roll={target_ori[0]:.1f}°, Pitch={target_ori[1]:.1f}°, Yaw={target_ori[2]:.1f}°")
            self.add_status_message("💡 미리보기 모드입니다. Run Simulation으로 실제 이동하세요.")
            
            self.target_position = old_target_pos
            self.target_orientation = old_target_ori
            
        except Exception as e:
            self.add_status_message(f"❌ 타겟 미리보기 오류: {str(e)}")
    
    def on_robot_type_change(self, event=None):
        """로봇 모드 변경 처리"""
        self.robot_type = self.type_var.get()

        if self.robot_type == "Inverse":
            self.calculate_btn.pack(side=tk.LEFT, padx=2, before=self.simulation_btn)
        else:
            self.calculate_btn.pack_forget()    
        
        self.clear_path_visualization()
        self.clear_ik_solutions()
        self.reset_simulation_state()  # 문제 4 해결
        self.update_joint_display()
        self.update_robot_display()
    
    def on_dof_change(self, event=None):
        """DOF 변경 처리"""
        new_dof = self.dof_var.get()
        if new_dof != self.current_dof:
            self.current_dof = new_dof
            
            if len(self.joint_angles) < new_dof:
                self.joint_angles.extend([0.0] * (new_dof - len(self.joint_angles)))
            else:
                self.joint_angles = self.joint_angles[:new_dof]
            
            self.load_default_dh_params_for_dof(new_dof)
            self.clear_path_visualization()
            self.clear_ik_solutions()
            self.reset_simulation_state()  # 문제 4 해결
            
            self.update_dh_parameter_inputs()
            self.update_joint_display()
            self.update_target_inputs_visibility()
            self.update_robot_display()
    
    def start_button_press(self, joint_idx, direction):
        """관절 각도 버튼 연속 누름 시작"""
        self.button_pressed[joint_idx] = True
        self.continuous_button_press(joint_idx, direction)
    
    def stop_button_press(self, joint_idx):
        """관절 각도 버튼 연속 누름 중지"""
        self.button_pressed[joint_idx] = False
    
    def continuous_button_press(self, joint_idx, direction):
        """관절 각도 버튼 연속 누름 처리"""
        if self.button_pressed.get(joint_idx, False):
            try:
                current_value = float(self.input_entries[joint_idx].get())
            except (ValueError, KeyError):
                current_value = 0.0
            
            new_value = current_value + direction * 1.0
            
            joint_limits = self.robot_kinematics.get_joint_limits(joint_idx)
            if joint_limits:
                min_limit, max_limit = joint_limits
                if new_value < min_limit or new_value > max_limit:
                    self.button_pressed[joint_idx] = False
                    return
            
            self.input_entries[joint_idx].delete(0, tk.END)
            self.input_entries[joint_idx].insert(0, f"{new_value:.2f}")
            self.joint_angles[joint_idx] = new_value
            
            self.update_robot_display()
            
            self.root.after(100, lambda: self.continuous_button_press(joint_idx, direction))
    
    # ========== 핵심 계산 및 시각화 기능 ==========
    
    def update_robot_display(self):
        """로봇 표시 업데이트 - 문제 3 해결: 정확한 위치 오차 추적"""
        try:
            current_dh_params = self.get_current_dh_params()
            
            if self.robot_type == "Forward":
                self.read_current_joint_angles()
            
            joint_angles_rad = [np.radians(angle) for angle in self.joint_angles]
            end_effector_T = self.robot_kinematics.forward_kinematics(current_dh_params, joint_angles_rad)
            
            position_m = end_effector_T[:3, 3]
            position_cm = position_m * self.M_TO_CM
            
            # 문제 3 해결: End-Effector 위치 정확 표시
            if hasattr(self, 'current_ee_label'):
                self.current_ee_label.config(
                    text=f"Current EE: X={position_cm[0]:.1f}, Y={position_cm[1]:.1f}, Z={position_cm[2]:.1f} cm"
                )
            
            # 문제 3 해결: 목표점과의 오차 계산 및 표시
            try:
                target_pos_cm = [float(self.target_pos_entries[i].get()) for i in range(3)]
                error_info = self.calculate_target_position_error(position_cm, target_pos_cm)
                
                if hasattr(self, 'position_error_label'):
                    error_color = 'darkgreen' if error_info['is_within_tolerance'] else 'darkred'
                    self.position_error_label.config(
                        text=f"Target Error: {error_info['total_error_mm']:.1f} mm (Tol: {self.position_tolerance_mm:.1f} mm)",
                        foreground=error_color
                    )
            except:
                if hasattr(self, 'position_error_label'):
                    self.position_error_label.config(text="Target Error: N/A")
            
            self.visualize_robot(current_dh_params, joint_angles_rad)
            
        except Exception as e:
            if hasattr(self, 'utils'):
                self.utils.log_message(f"Robot display update error: {e}", "ERROR")
    
    def read_current_joint_angles(self):
        """현재 관절 각도 읽기"""
        if self.robot_type == "Forward":
            for i in range(self.current_dof):
                try:
                    if i in self.input_entries:
                        angle = float(self.input_entries[i].get())
                        self.joint_angles[i] = angle
                except (ValueError, KeyError):
                    self.joint_angles[i] = 0.0
    
    def get_current_dh_params(self):
        """현재 GUI에서 DH 파라미터 읽어오기"""
        current_params = []
        
        for i in range(self.current_dof):
            try:
                a = float(self.dh_entries[i]['a'].get())
                alpha = float(self.dh_entries[i]['alpha'].get())
                d = float(self.dh_entries[i]['d'].get())
                theta = float(self.dh_entries[i]['theta'].get())
                current_params.append([a, alpha, d, theta])
            except (ValueError, KeyError):
                current_params.append([20.0, 0.0, 0.0, 0.0])
        
        return current_params
    
    def calculate_inverse_kinematics(self):
        """Inverse Kinematics 계산"""
        if self.robot_type != "Inverse":
            return
        
        try:
            dh_params = self.get_current_dh_params()
            
            target_pos_cm = []
            for i in range(3):
                try:
                    pos = float(self.target_pos_entries[i].get())
                    target_pos_cm.append(pos)
                except (ValueError, KeyError):
                    target_pos_cm.append(0.0)
            
            target_pos_m = [pos * self.CM_TO_M for pos in target_pos_cm]
            
            target_ori = None
            if self.current_dof >= 3:
                try:
                    target_ori = []
                    for i in range(3):
                        ori = float(self.target_ori_entries[i].get())
                        target_ori.append(np.radians(ori))
                except:
                    target_ori = [0.0, 0.0, 0.0]
            
            solutions = self.find_multiple_ik_solutions(dh_params, target_pos_m, target_ori)
            
            self.display_ik_results(solutions, target_pos_cm, target_ori)
            
        except Exception as e:
            if self.result_text is not None:
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(1.0, f"IK 계산 오류: {str(e)}")
    
    def find_multiple_ik_solutions(self, dh_params, target_pos_m, target_ori):
        """다중 IK 해 탐색"""
        solutions = []
        
        initial_guesses = self.generate_initial_guesses()
        
        for initial_guess in initial_guesses:
            try:
                solution = self.robot_kinematics.inverse_kinematics(
                    dh_params, target_pos_m, target_ori, initial_guess, method='numerical'
                )
                
                if solution is not None:
                    if self.validate_ik_solution(solution, dh_params, target_pos_m, target_ori, self.position_tolerance_mm / 1000):
                        is_duplicate = False
                        for existing_sol in solutions:
                            angle_diff = np.array([abs(np.degrees(s1 - s2)) for s1, s2 in zip(solution, existing_sol)])
                            if np.all(angle_diff < 5.0):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            solutions.append(solution)
                            
                    if len(solutions) >= 3:
                        break
                        
            except Exception as e:
                continue
        
        return solutions
    
    def generate_initial_guesses(self):
        """초기 추정값 생성"""
        initial_guesses = []
        
        current_angles_rad = [np.radians(angle) for angle in self.joint_angles]
        initial_guesses.append(current_angles_rad)
        
        initial_guesses.append([0.0] * self.current_dof)
        
        if self.current_dof >= 2:
            elbow_up = [0.0] * self.current_dof
            elbow_up[1] = np.radians(90)
            initial_guesses.append(elbow_up)
            
            elbow_down = [0.0] * self.current_dof
            elbow_down[1] = np.radians(-90)
            initial_guesses.append(elbow_down)
        
        for _ in range(3):
            random_angles = [np.radians(np.random.uniform(-90, 90)) for _ in range(self.current_dof)]
            initial_guesses.append(random_angles)
        
        return initial_guesses
    
    def validate_ik_solution(self, solution, dh_params, target_pos, target_ori, tolerance):
        """IK 해의 유효성 검증"""
        try:
            verify_T = self.robot_kinematics.forward_kinematics(dh_params, solution)
            verify_pos = verify_T[:3, 3]
            
            pos_error = np.linalg.norm(np.array(target_pos) - verify_pos)
            if pos_error > tolerance:
                return False
            
            for i, angle in enumerate(solution):
                joint_limits = self.robot_kinematics.get_joint_limits(i)
                if joint_limits:
                    min_limit, max_limit = joint_limits
                    angle_deg = np.degrees(angle)
                    if angle_deg < min_limit or angle_deg > max_limit:
                        return False
            
            return True
            
        except Exception as e:
            return False
    
    def display_ik_results(self, solutions, target_pos_cm, target_ori):
        """IK 결과 표시"""
        self.ik_solutions = solutions
        
        result_text = f"=== Inverse Kinematics 결과 ===\n"
        result_text += f"DOF: {self.current_dof}\n"
        result_text += f"목표 위치: X={target_pos_cm[0]:.2f}, Y={target_pos_cm[1]:.2f}, Z={target_pos_cm[2]:.2f} cm\n"
        
        if target_ori:
            ori_deg = [np.degrees(ori) for ori in target_ori]
            result_text += f"목표 자세: Roll={ori_deg[0]:.2f}°, Pitch={ori_deg[1]:.2f}°, Yaw={ori_deg[2]:.2f}°\n"
        
        result_text += f"\n발견된 해의 개수: {len(solutions)}\n"
        
        if solutions:
            for i, solution in enumerate(solutions):
                solution_deg = [np.degrees(angle) for angle in solution]
                result_text += f"해 {i+1}: "
                result_text += ", ".join([f"J{j+1}:{angle:.1f}°" for j, angle in enumerate(solution_deg)])
                result_text += "\n"
            
            self.create_ik_solution_buttons()
            self.apply_ik_solution_to_display(0)
            
        else:
            result_text += "해를 찾을 수 없습니다.\n"
            result_text += self.analyze_ik_failure()
        
        if self.result_text is not None:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(1.0, result_text)
    
    def create_ik_solution_buttons(self):
        """IK 해 선택 버튼들 생성"""
        # 기존 버튼들 제거
        for widget in self.ik_solutions_scrollable_frame.winfo_children():
            widget.destroy()
        
        # 프레임 표시
        self.ik_solutions_frame.pack(fill=tk.X, pady=5)
        
        title_label = ttk.Label(self.ik_solutions_scrollable_frame, 
                               text="Available Solutions (Click to Select):", 
                               font=('Arial', 10, 'bold'))
        title_label.pack(anchor="w", padx=5, pady=2)
        
        for i, solution in enumerate(self.ik_solutions):
            solution_deg = [np.degrees(angle) for angle in solution]
            angles_text = ", ".join([f"J{j+1}:{angle:.1f}°" for j, angle in enumerate(solution_deg)])
            button_text = f"Sol {i+1}: {angles_text}"
            
            btn = ttk.Button(self.ik_solutions_scrollable_frame, text=button_text, 
                           command=lambda idx=i: self.select_ik_solution(idx))
            btn.pack(fill=tk.X, padx=5, pady=1)
        
        # 스크롤 영역 업데이트
        self.ik_solutions_scrollable_frame.update_idletasks()
        self.ik_solutions_canvas.configure(scrollregion=self.ik_solutions_canvas.bbox("all"))
    
    def select_ik_solution(self, solution_index):
        """IK 해 선택"""
        if 0 <= solution_index < len(self.ik_solutions):
            self.selected_ik_solution = solution_index
            self.apply_ik_solution_to_display(solution_index)
    
    def apply_ik_solution_to_display(self, solution_index):
        """선택된 IK 해를 표시에 적용"""
        if 0 <= solution_index < len(self.ik_solutions):
            solution = self.ik_solutions[solution_index]
            dh_params = self.get_current_dh_params()
            self.visualize_robot(dh_params, solution)
    
    def visualize_robot(self, dh_params, joint_angles):
        """로봇 시각화"""
        try:
            self.ax.clear()
            
            link_positions = self.visualizer.compute_link_positions(dh_params, joint_angles)
            
            link_positions_m = []
            for pos in link_positions:
                pos_m = [pos[0] * self.CM_TO_M, pos[1] * self.CM_TO_M, pos[2] * self.CM_TO_M]
                link_positions_m.append(pos_m)
            
            self.visualizer.draw_robot_links(self.ax, link_positions_m)
            self.visualizer.draw_joints(self.ax, link_positions_m)
            
            try:
                target_pos_cm = [float(self.target_pos_entries[i].get()) for i in range(3)]
                target_pos_m = [pos * self.CM_TO_M for pos in target_pos_cm]
                self.ax.scatter(target_pos_m[0], target_pos_m[1], target_pos_m[2], 
                               c='red', s=150, marker='*', alpha=0.8, label='Target')
            except:
                pass
            
            if self.current_path is not None and len(self.current_path) > 0:
                path_array = np.array(self.current_path)
                self.ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                           'g--', linewidth=2, alpha=0.7, label='Planned Path')
            
            if self.trajectory_history and len(self.trajectory_history) > 1:
                history_array = np.array(self.trajectory_history)
                self.ax.plot(history_array[:, 0], history_array[:, 1], history_array[:, 2],
                           'b-', linewidth=1, alpha=0.5, label='Movement History')
            
            max_reach = sum([abs(param[0]) for param in dh_params]) * self.CM_TO_M * 1.2
            limit = max(max_reach, 0.8)
            
            self.ax.set_xlim(-limit, limit)
            self.ax.set_ylim(-limit, limit)
            self.ax.set_zlim(0.0, limit * 1.5)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f"{self.current_dof}DOF Robot - {self.robot_type} Mode")
            
            self.ax.grid(True, alpha=0.3)
            
            handles, labels = self.ax.get_legend_handles_labels()
            if handles:
                self.ax.legend()
            
            self.canvas.draw()
            
        except Exception as e:
            if hasattr(self, 'utils'):
                self.utils.log_message(f"시각화 오류: {e}", "ERROR")
            self.setup_robot_visualization()
    
    def analyze_ik_failure(self):
        """IK 실패 원인 분석"""
        analysis_text = "\n=== 실패 원인 분석 ===\n"
        
        dh_params = self.get_current_dh_params()
        target_pos_cm = [float(self.target_pos_entries[i].get()) for i in range(3)]
        target_pos_m = [pos * self.CM_TO_M for pos in target_pos_cm]
        
        max_reach = sum([abs(param[0]) for param in dh_params]) * self.CM_TO_M
        target_distance = np.linalg.norm(target_pos_m)
        
        analysis_text += f"최대 도달거리: {max_reach:.3f} m ({max_reach * self.M_TO_CM:.1f} cm)\n"
        analysis_text += f"목표까지 거리: {target_distance:.3f} m ({target_distance * self.M_TO_CM:.1f} cm)\n"
        
        if target_distance > max_reach * 0.95:
            analysis_text += "❌ 목표 위치가 작업공간을 벗어남\n"
            analysis_text += "💡 해결방안: 목표 위치를 가까이 이동하거나 링크 길이 증가\n"
        elif target_distance < max_reach * 0.1:
            analysis_text += "❌ 목표 위치가 너무 가까움 (특이점 영역)\n" 
            analysis_text += "💡 해결방안: 목표 위치를 더 멀리 이동\n"
        else:
            analysis_text += "⚠️ 목표는 도달 범위 내이지만 다른 제약으로 인해 해가 없음\n"
            analysis_text += "💡 해결방안: 목표 자세 변경 또는 관절 제한 확인\n"
        
        return analysis_text
    
    # ========== 문제 3, 4 해결: 개선된 시뮬레이션 기능 ==========
    
    def run_goal_oriented_simulation(self):
        """목표 지향적 시뮬레이션 실행 - 문제 3, 4 해결"""
        if self.simulation_running:
            self.add_status_message("⚠️ 시뮬레이션이 이미 실행 중입니다.")
            return
        
        try:
            # 문제 4 해결: 매번 시각적 상태 초기화
            if not self.simulation_state['is_first_run']:
                self.reset_simulation_state()
            
            simulation_plan = self.create_goal_oriented_plan()
            if simulation_plan is None:
                return
            
            path_result = self.plan_path_to_goal(simulation_plan)
            if not path_result['success']:
                if self.robot_type == "Forward":
                    self.add_status_message(f"❌ 목표 위치로의 경로 계획 실패: {path_result['reason']}")
                else:
                    self.add_status_message(f"❌ 경로 계획 실패: {path_result['reason']}")
                return
            
            self.simulation_running = True
            self.enable_controls(False)
            
            self.add_status_message("🎯 목표 지향적 시뮬레이션 시작...")
            
            # 문제 3 해결: 정확한 목표점 추적을 위한 초기 상태 기록
            initial_ee_pos = self.get_current_ee_position()
            target_pos_cm = [float(self.target_pos_entries[i].get()) for i in range(3)]
            
            self.add_status_message(f"📍 시작 위치: X={initial_ee_pos[0]:.1f}, Y={initial_ee_pos[1]:.1f}, Z={initial_ee_pos[2]:.1f} cm")
            self.add_status_message(f"🎯 목표 위치: X={target_pos_cm[0]:.1f}, Y={target_pos_cm[1]:.1f}, Z={target_pos_cm[2]:.1f} cm")
            
            self.execute_goal_simulation_enhanced(path_result)
            
        except Exception as e:
            self.simulation_running = False
            self.enable_controls(True)
            self.add_status_message(f"❌ 시뮬레이션 오류: {str(e)}")
    
    def get_current_ee_position(self):
        """현재 End-Effector 위치 반환 (cm 단위)"""
        try:
            dh_params = self.get_current_dh_params()
            joint_angles_rad = [np.radians(angle) for angle in self.joint_angles]
            end_effector_T = self.robot_kinematics.forward_kinematics(dh_params, joint_angles_rad)
            position_m = end_effector_T[:3, 3]
            return position_m * self.M_TO_CM
        except:
            return [0.0, 0.0, 0.0]
    
    def create_goal_oriented_plan(self):
        """목표 지향적 시뮬레이션 계획 생성"""
        try:
            dh_params = self.get_current_dh_params()
            
            start_angles = [np.radians(angle) for angle in self.joint_angles]
            
            target_pos_cm = []
            for i in range(3):
                try:
                    pos = float(self.target_pos_entries[i].get())
                    target_pos_cm.append(pos)
                except (ValueError, KeyError):
                    target_pos_cm.append(0.0)
            
            target_pos_m = [pos * self.CM_TO_M for pos in target_pos_cm]
            
            target_ori = None
            if self.current_dof >= 3:
                try:
                    target_ori = []
                    for i in range(3):
                        ori = float(self.target_ori_entries[i].get())
                        target_ori.append(np.radians(ori))
                except:
                    target_ori = [0.0, 0.0, 0.0]
            
            if self.robot_type == "Inverse" and self.ik_solutions:
                target_angles = self.ik_solutions[self.selected_ik_solution]
            else:
                target_angles = self.robot_kinematics.inverse_kinematics(
                    dh_params, target_pos_m, target_ori, start_angles, method='numerical'
                )
                
                if target_angles is None:
                    if self.robot_type == "Forward":
                        self.add_status_message("❌ 목표 위치로 이동하는 경로를 계획할 수 없습니다.")
                        self.add_status_message("💡 목표 위치가 로봇의 작업공간 내에 있는지 확인하세요.")
                    else:
                        self.add_status_message("❌ 목표 위치에 대한 IK 해를 찾을 수 없습니다.")
                    return None
            
            plan = {
                'dh_params': dh_params,
                'start_angles': start_angles,
                'target_angles': target_angles,
                'target_position_m': target_pos_m,
                'target_position_cm': target_pos_cm,
                'mode': self.robot_type
            }
            
            return plan
            
        except Exception as e:
            self.add_status_message(f"❌ 시뮬레이션 계획 생성 오류: {str(e)}")
            return None
    
    def plan_path_to_goal(self, plan):
        """목표까지의 경로 계획"""
        try:
            start_angles = plan['start_angles']
            target_angles = plan['target_angles']
            dh_params = plan['dh_params']
            
            n_steps = 40
            trajectory = []
            
            for i in range(n_steps + 1):
                t = i / n_steps
                current_angles = []
                
                for j in range(len(start_angles)):
                    angle = self.utils.interpolate_angles(start_angles[j], target_angles[j], t)
                    current_angles.append(angle)
                
                trajectory.append(current_angles)
            
            validation_result = self.validate_trajectory_path(trajectory, dh_params)
            
            if validation_result['valid']:
                end_effector_path = []
                for joint_angles in trajectory:
                    T = self.robot_kinematics.forward_kinematics(dh_params, joint_angles)
                    end_effector_path.append(T[:3, 3].tolist())
                
                self.current_path = end_effector_path
                
                return {
                    'success': True,
                    'trajectory': trajectory,
                    'end_effector_path': end_effector_path,
                    'warnings': validation_result.get('warnings', [])
                }
            else:
                return {
                    'success': False,
                    'reason': validation_result['reason'],
                    'trajectory': None
                }
            
        except Exception as e:
            return {
                'success': False,
                'reason': f"경로 계획 중 오류: {str(e)}",
                'trajectory': None
            }
    
    def validate_trajectory_path(self, trajectory, dh_params):
        """궤적 경로 유효성 검사"""
        try:
            warnings = []
            
            for i, joint_angles in enumerate(trajectory):
                for j, angle in enumerate(joint_angles):
                    joint_limits = self.robot_kinematics.get_joint_limits(j)
                    if joint_limits:
                        min_limit, max_limit = joint_limits
                        angle_deg = np.degrees(angle)
                        if angle_deg < min_limit or angle_deg > max_limit:
                            return {
                                'valid': False,
                                'reason': f"관절 {j+1}이 제한을 초과 (스텝 {i+1}: {angle_deg:.1f}° ∉ [{min_limit}°, {max_limit}°])",
                                'warnings': warnings
                            }
                
                try:
                    link_positions = self.visualizer.compute_link_positions(dh_params, joint_angles)
                    for pos in link_positions:
                        z_pos_m = pos[2] * self.CM_TO_M
                        if z_pos_m < -0.02:
                            return {
                                'valid': False,
                                'reason': f"스텝 {i+1}에서 바닥과 충돌 (Z={z_pos_m:.3f}m)",
                                'warnings': warnings
                            }
                except:
                    pass
                
                try:
                    jacobian = self.robot_kinematics.compute_jacobian(dh_params, joint_angles)
                    singularity_info = self.robot_kinematics.check_singularity(jacobian)
                    
                    if singularity_info['is_singular']:
                        warnings.append(f"스텝 {i+1}에서 특이점 근처")
                        
                except:
                    pass
            
            return {
                'valid': True,
                'reason': '',
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'valid': False,
                'reason': f"경로 검증 중 오류: {str(e)}",
                'warnings': []
            }
    
    def execute_goal_simulation_enhanced(self, path_result):
        """목표 시뮬레이션 실행 - 문제 3, 4 해결: 정확성 추적 및 시각적 일관성"""
        trajectory = path_result['trajectory']
        
        def animate_step(step):
            if not self.simulation_running or step >= len(trajectory):
                self.simulation_running = False
                self.enable_controls(True)
                
                if step > 0:
                    final_angles = trajectory[-1]
                    final_angles_deg = [np.degrees(angle) for angle in final_angles]
                    self.joint_angles = final_angles_deg
                    
                    if self.robot_type == "Forward":
                        for i, angle_deg in enumerate(final_angles_deg):
                            if i in self.input_entries:
                                self.input_entries[i].delete(0, tk.END)
                                self.input_entries[i].insert(0, f"{angle_deg:.2f}")
                    else:
                        self.update_joint_display()
                
                # 문제 3 해결: 정확한 최종 위치 분석 및 오차 추적
                self.analyze_final_position_accuracy_enhanced(trajectory[-1] if trajectory else None)
                
                # 문제 4 해결: 시뮬레이션 상태 업데이트
                self.simulation_state['is_first_run'] = False
                self.simulation_state['last_joint_angles'] = self.joint_angles.copy()
                
                if path_result.get('warnings'):
                    self.add_status_message("⚠️ 경고사항:")
                    for warning in path_result['warnings']:
                        self.add_status_message(f"  • {warning}")
                
                return
            
            current_joint_angles = trajectory[step]
            
            try:
                dh_params = self.get_current_dh_params()
                T = self.robot_kinematics.forward_kinematics(dh_params, current_joint_angles)
                ee_pos = T[:3, 3].tolist()
                self.trajectory_history.append(ee_pos)
            except:
                pass
            
            self.visualize_robot(self.get_current_dh_params(), current_joint_angles)
            
            # 진행률 표시 (문제 1 해결: 줄간격 없이)
            progress = (step / len(trajectory)) * 100
            if step % 5 == 0:  # 5스텝마다 업데이트
                self.add_status_message(f"진행: {progress:.1f}% ({step+1}/{len(trajectory)})")
            
            self.root.after(100, lambda: animate_step(step + 1))
        
        self.trajectory_history = []
        animate_step(0)
    
    def analyze_final_position_accuracy_enhanced(self, final_joint_angles):
        """문제 3 해결: 향상된 최종 위치 정확도 분석"""
        if final_joint_angles is None:
            self.add_status_message("❌ 시뮬레이션 데이터가 없습니다.")
            return
        
        try:
            # 실제 도달한 위치 계산
            dh_params = self.get_current_dh_params()
            final_T = self.robot_kinematics.forward_kinematics(dh_params, final_joint_angles)
            actual_pos_m = final_T[:3, 3]
            actual_pos_cm = actual_pos_m * self.M_TO_CM
            
            # 목표 위치
            target_pos_cm = []
            for i in range(3):
                try:
                    pos = float(self.target_pos_entries[i].get())
                    target_pos_cm.append(pos)
                except:
                    target_pos_cm.append(0.0)
            
            # 문제 3 해결: 정확한 오차 분석
            error_info = self.calculate_target_position_error(actual_pos_cm, target_pos_cm)
            
            # 결과 표시
            self.add_status_message("=== 시뮬레이션 완료 ===")
            self.add_status_message(f"📍 실제 도달 위치: X={actual_pos_cm[0]:.2f}, Y={actual_pos_cm[1]:.2f}, Z={actual_pos_cm[2]:.2f} cm")
            self.add_status_message(f"🎯 목표 위치: X={target_pos_cm[0]:.2f}, Y={target_pos_cm[1]:.2f}, Z={target_pos_cm[2]:.2f} cm")
            
            if error_info['is_within_tolerance']:
                self.add_status_message("✅ 목표 위치에 성공적으로 도달했습니다!")
                self.add_status_message(f"📏 총 오차: {error_info['total_error_mm']:.1f} mm (허용범위: {self.position_tolerance_mm:.1f} mm)")
            else:
                self.add_status_message("⚠️ 목표 위치에 완전히 도달하지 못했습니다.")
                self.add_status_message(f"❌ 총 오차: {error_info['total_error_mm']:.1f} mm (허용범위 초과)")
                self.add_status_message(f"   X축 오차: {error_info['x_error_mm']:.1f} mm")
                self.add_status_message(f"   Y축 오차: {error_info['y_error_mm']:.1f} mm")
                self.add_status_message(f"   Z축 오차: {error_info['z_error_mm']:.1f} mm")
                
                # 문제 3 해결: 정확한 해결방안 제시
                if error_info['total_error_mm'] > 20:
                    self.add_status_message("💡 해결방안: 목표 위치가 작업공간을 벗어났을 가능성 - 더 가까운 목표점 설정")
                elif error_info['total_error_mm'] > 10:
                    self.add_status_message("💡 해결방안: IK 알고리즘 정밀도 개선 필요 또는 허용 오차 조정")
                else:
                    self.add_status_message("💡 해결방안: 경미한 오차 - 허용 오차 범위 조정 검토")
        
        except Exception as e:
            self.add_status_message(f"❌ 위치 정확도 분석 오류: {str(e)}")
    
    def enable_controls(self, enabled=True):
        """컨트롤 활성화/비활성화"""
        state = 'normal' if enabled else 'disabled'
        
        for widget in self.control_widgets:
            try:
                self._set_widget_state(widget, state)
            except:
                continue
        
        if hasattr(self, 'simulation_btn'):
            self.simulation_btn.config(state=state)
    
    def _set_widget_state(self, widget, state):
        """위젯 상태 설정"""
        try:
            if hasattr(widget, 'config'):
                widget.config(state=state)
        except:
            pass
        
        try:
            for child in widget.winfo_children():
                self._set_widget_state(child, state)
        except:
            pass
    
    def clear_path_visualization(self):
        """경로 시각화 초기화"""
        self.current_path = None
        self.trajectory_history = []
    
    def clear_ik_solutions(self):
        """IK 해 초기화"""
        self.ik_solutions = []
        self.selected_ik_solution = 0
        
        # IK Solutions 프레임 숨기기
        self.ik_solutions_frame.pack_forget()
        
        # 스크롤 가능한 프레임 내용 지우기
        for widget in self.ik_solutions_scrollable_frame.winfo_children():
            widget.destroy()
    
    # ========== 기타 기능들 (간략화) ==========
    
    def load_default_dh_params_for_dof(self, dof):
        """기본 DH 파라미터 로드"""
        self.dh_params = self.dh_manager.get_default_dh_params(dof)
    
    def load_initial_configuration(self):
        """초기 구성 로드"""
        try:
            self.load_default_dh_params_for_dof(1)
            self.update_dh_parameter_inputs()
            self.update_joint_display()
            self.update_robot_display()
        except Exception as e:
            if hasattr(self, 'utils'):
                self.utils.log_message(f"초기 구성 로드 오류: {e}", "ERROR")
    
    def generate_random_parameters(self):
        """랜덤 파라미터 생성"""
        try:
            random_params = self.dh_manager.generate_random_params(
                self.current_dof, 
                link_length_range=(15, 40),
                angle_range=(-90, 90)
            )
            
            for i, params in enumerate(random_params):
                if i < len(self.dh_entries):
                    param_names = ['a', 'alpha', 'd', 'theta']
                    for j, param_name in enumerate(param_names):
                        if param_name in self.dh_entries[i]:
                            self.dh_entries[i][param_name].delete(0, tk.END)
                            self.dh_entries[i][param_name].insert(0, f"{params[j]:.1f}")
            
            self.clear_path_visualization()
            self.clear_ik_solutions()
            self.reset_simulation_state()  # 문제 4 해결
            self.update_robot_display()
            
        except Exception as e:
            self.add_status_message(f"랜덤 파라미터 생성 오류: {str(e)}")
    
    def reset_to_default(self):
        """기본값으로 초기화"""
        try:
            self.current_dof = 1
            self.robot_type = "Forward"
            self.joint_angles = [0.0]
            self.target_position = [30.0, 0.0, 15.0]
            self.target_orientation = [0.0, 0.0, 0.0]
            
            self.dof_var.set(1)
            self.type_var.set("Forward")
            
            self.load_default_dh_params_for_dof(1)
            self.clear_path_visualization()
            self.clear_ik_solutions()
            self.reset_simulation_state()  # 문제 4 해결
            
            self.update_dh_parameter_inputs()
            self.update_joint_display()
            self.update_target_inputs_visibility()
            
            for i in range(3):
                if i in self.target_pos_entries:
                    self.target_pos_entries[i].delete(0, tk.END)
                    self.target_pos_entries[i].insert(0, f"{self.target_position[i]:.2f}")
            
            for i in range(3):
                if i in self.target_ori_entries:
                    self.target_ori_entries[i].delete(0, tk.END)
                    self.target_ori_entries[i].insert(0, f"{self.target_orientation[i]:.2f}")
            
            self.update_robot_display()
            
            self.add_status_message("✅ 기본 1DOF 구성으로 초기화되었습니다.")
            
        except Exception as e:
            self.add_status_message(f"초기화 오류: {str(e)}")
    
    def analyze_workspace(self):
        """작업공간 분석"""
        try:
            self.add_status_message("=== 작업공간 분석 ===")
            self.add_status_message("분석 중...")
            
            dh_params = self.get_current_dh_params()
            workspace_points = self.robot_kinematics.compute_workspace(dh_params, resolution=25)
            
            if len(workspace_points) > 0:
                distances = np.sqrt(workspace_points[:, 0]**2 + workspace_points[:, 1]**2 + workspace_points[:, 2]**2)
                max_reach = np.max(distances)
                min_reach = np.min(distances)
                
                x_range = [np.min(workspace_points[:, 0]), np.max(workspace_points[:, 0])]
                y_range = [np.min(workspace_points[:, 1]), np.max(workspace_points[:, 1])]
                z_range = [np.min(workspace_points[:, 2]), np.max(workspace_points[:, 2])]
                
                self.add_status_message("작업공간 통계:")
                self.add_status_message(f"  최대 도달거리: {max_reach:.3f} m ({max_reach * 100:.1f} cm)")
                self.add_status_message(f"  최소 도달거리: {min_reach:.3f} m ({min_reach * 100:.1f} cm)")
                self.add_status_message(f"  X 범위: {x_range[0]:.3f} ~ {x_range[1]:.3f} m")
                self.add_status_message(f"  Y 범위: {y_range[0]:.3f} ~ {y_range[1]:.3f} m")
                self.add_status_message(f"  Z 범위: {z_range[0]:.3f} ~ {z_range[1]:.3f} m")
                self.add_status_message(f"  샘플 포인트: {len(workspace_points)}개")
            else:
                self.add_status_message("작업공간 계산 실패")
                    
        except Exception as e:
            self.add_status_message(f"작업공간 분석 오류: {str(e)}")
    
    def save_results(self):
        """결과 저장"""
        try:
            results_dir = "./results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_simulation_{self.current_dof}DOF_{timestamp}.csv"
            filepath = os.path.join(results_dir, filename)
            
            data = {
                'DOF': self.current_dof,
                'Robot_Mode': self.robot_type,
                'Joint_Angles_deg': self.joint_angles,
                'Target_Position_cm': [float(self.target_pos_entries[i].get()) for i in range(3)],
                'End_Effector_Position_cm': self.get_current_ee_position(),
                'DH_Parameters': self.get_current_dh_params()
            }
            
            if self.ik_solutions:
                data['IK_Solutions'] = [[np.degrees(angle) for angle in sol] for sol in self.ik_solutions]
            
            if self.current_path:
                data['Planned_Path'] = self.current_path
            
            if self.trajectory_history:
                data['Movement_History'] = self.trajectory_history
            
            self.utils.save_results_to_csv(data, filepath)
            
            self.add_status_message(f"💾 결과 저장: {os.path.basename(filepath)}")
            
        except Exception as e:
            self.add_status_message(f"저장 오류: {str(e)}")
    
    def load_yaml_parameters(self):
        """YAML 파일 로드"""
        file_path = filedialog.askopenfilename(
            title="Load DH Parameters",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir="./yaml"
        )
        
        if file_path:
            try:
                params = self.dh_manager.load_from_yaml(file_path)
                if params:
                    new_dof = len(params)
                    self.current_dof = new_dof
                    self.dof_var.set(new_dof)
                    
                    self.joint_angles = [0.0] * new_dof
                    self.dh_params = params
                    
                    self.clear_path_visualization()
                    self.clear_ik_solutions()
                    self.reset_simulation_state()
                    
                    self.update_dh_parameter_inputs()
                    self.update_joint_display()
                    self.update_target_inputs_visibility()
                    
                    for i, param in enumerate(params):
                        if i < len(self.dh_entries):
                            param_names = ['a', 'alpha', 'd', 'theta']
                            for j, param_name in enumerate(param_names):
                                if param_name in self.dh_entries[i]:
                                    self.dh_entries[i][param_name].delete(0, tk.END)
                                    self.dh_entries[i][param_name].insert(0, f"{param[j]:.1f}")
                    
                    self.update_robot_display()
                    
            except Exception as e:
                self.add_status_message(f"YAML 로드 오류: {str(e)}")
    
    def save_yaml_parameters(self):
        """YAML 파일 저장"""
        try:
            dh_params = self.get_current_dh_params()
            
            file_path = filedialog.asksaveasfilename(
                title="Save DH Parameters",
                defaultextension=".yaml",
                filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
                initialdir="./yaml"
            )
            
            if file_path:
                self.dh_manager.save_to_yaml(dh_params, file_path, self.current_dof)
                self.add_status_message(f"YAML 저장: {os.path.basename(file_path)}")
                
        except Exception as e:
            self.add_status_message(f"YAML 저장 오류: {str(e)}")
    
    def load_preset_robot(self):
        """프리셋 로봇 로드"""
        presets = {
            "1DOF Simple": (1, [[30.0, 0.0, 0.0, 0.0]]),
            "2DOF Planar": (2, [[40.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]]),
            "3DOF Anthropomorphic": (3, [[0.0, 90.0, 15.0, 0.0], [35.0, 0.0, 0.0, 0.0], [25.0, 0.0, 0.0, 0.0]]),
            "6DOF Industrial": (6, [[0.0, 90.0, 15.0, 0.0], [25.0, 0.0, 0.0, 0.0], [3.0, 90.0, 0.0, 0.0], 
                                   [0.0, -90.0, 22.0, 0.0], [0.0, 90.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
        }
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Preset Robot")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Choose a preset robot configuration:", 
                 font=('Arial', 12)).pack(pady=10)
        
        listbox = tk.Listbox(dialog, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for preset_name in presets.keys():
            listbox.insert(tk.END, preset_name)
        
        def load_selected():
            selection = listbox.curselection()
            if selection:
                preset_name = list(presets.keys())[selection[0]]
                dof, params = presets[preset_name]
                
                self.current_dof = dof
                self.dof_var.set(dof)
                self.joint_angles = [0.0] * dof
                self.dh_params = params
                
                self.clear_path_visualization()
                self.clear_ik_solutions()
                self.reset_simulation_state()
                
                self.update_dh_parameter_inputs()
                self.update_joint_display()
                self.update_target_inputs_visibility()
                self.update_robot_display()
                
                dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Load", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)


def main():
    """메인 함수"""
    print("=" * 70)
    print("Robot Kinematics Simulation - 핵심 문제 해결 버전")
    print("4개 핵심 문제 완전 해결")
    print("=" * 70)
    
    folders = ["./yaml", "./results"]
    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"✓ 폴더 확인: {folder}")
        except Exception as e:
            print(f"✗ 폴더 생성 오류 {folder}: {e}")
    
    try:
        print("\n해결된 핵심 문제들:")
        print("1. ✅ 상태창 텍스트 줄간격 문제 완전 해결")
        print("   - 불필요한 공백 줄 제거")
        print("   - 깔끔한 로그 메시지 출력")
        print("   - 연속된 줄바꿈 통합")
        
        print("2. ✅ Auto Update 코드 완전 제거")
        print("   - Auto Update 관련 모든 변수 및 메서드 제거")
        print("   - 직관적인 'Calculate' 버튼으로 대체")
        print("   - 명시적 Forward Kinematics 계산")
        
        print("3. ✅ 목표점 위치와 시뮬레이션 정확성 추적 시스템 구축")
        print("   - 정확한 오차 계산 및 표시 (mm 단위)")
        print("   - End-Effector와 목표점 간 실시간 오차 모니터링")
        print("   - 허용 오차 기반 성공/실패 판정")
        print("   - 축별 상세 오차 분석")
        
        print("4. ✅ 시뮬레이션 초기화 및 시각적 동작 일관성 보장")
        print("   - 매 실행마다 시각적 상태 완전 초기화")
        print("   - 일관된 애니메이션 제공")
        print("   - 시뮬레이션 상태 추적 시스템")
        print("   - 재실행 시에도 동일한 시각적 효과")
        
        print("\n추가 개선사항:")
        print("• 허용 오차를 5mm로 설정하여 정밀도 향상")
        print("• 실시간 위치 오차 표시 라벨 추가")
        print("• 목표점 추적을 위한 정밀한 분석 시스템")
        print("• 사용자 친화적인 오류 메시지 및 해결방안 제시")
        print("=" * 70)
        
        root = tk.Tk()
        app = RobotSimulationGUI(root)
        print("✓ GUI 초기화 완료 (핵심 문제 해결 버전)")
        
        root.mainloop()
        
    except Exception as e:
        print(f"✗ GUI 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()