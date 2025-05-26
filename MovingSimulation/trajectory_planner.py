"""
trajectory_planner.py - MovingSimulation/trajectory_planner.py

궤적 계획 모듈
- Quintic (5차) 다항식 궤적 생성
- 각도, 각속도, 각가속도 프로파일 생성
- 저크(Jerk) 최소화 궤적
- 그래프 생성 및 저장 기능
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

class TrajectoryPlanner:
    def __init__(self):
        """궤적 계획기 초기화"""
        # 기본 설정값들
        self.default_duration = 5.0  # 기본 궤적 시간 (초)
        self.default_dt = 0.01       # 기본 시간 간격 (초)
        
        # 그래프 스타일 설정
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plan_quintic_trajectory(self, start_angles, end_angles, duration=None, dt=None,
                               start_velocities=None, end_velocities=None,
                               start_accelerations=None, end_accelerations=None):
        """
        5차 다항식을 이용한 궤적 계획
        
        5차 다항식 형태: θ(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        
        경계 조건:
        - θ(0) = θ_start, θ(T) = θ_end
        - θ'(0) = ω_start, θ'(T) = ω_end  
        - θ''(0) = α_start, θ''(T) = α_end
        
        Args:
            start_angles (list): 시작 관절 각도 (라디안)
            end_angles (list): 끝 관절 각도 (라디안)
            duration (float): 궤적 지속 시간 (초)
            dt (float): 시간 간격 (초)
            start_velocities (list): 시작 각속도 (라디안/초, 기본값: 0)
            end_velocities (list): 끝 각속도 (라디안/초, 기본값: 0)
            start_accelerations (list): 시작 각가속도 (라디안/초², 기본값: 0)
            end_accelerations (list): 끝 각가속도 (라디안/초², 기본값: 0)
            
        Returns:
            dict: 궤적 데이터
                - time: 시간 배열
                - positions: 각도 배열 [time_steps, n_joints]
                - velocities: 각속도 배열 [time_steps, n_joints]
                - accelerations: 각가속도 배열 [time_steps, n_joints]
                - jerks: 저크 배열 [time_steps, n_joints]
                - coefficients: 다항식 계수 [n_joints, 6]
        """
        # 기본값 설정
        if duration is None:
            duration = self.default_duration
        if dt is None:
            dt = self.default_dt
            
        n_joints = len(start_angles)
        
        # 기본 속도와 가속도 설정 (0으로 시작하고 끝남)
        if start_velocities is None:
            start_velocities = [0.0] * n_joints
        if end_velocities is None:
            end_velocities = [0.0] * n_joints
        if start_accelerations is None:
            start_accelerations = [0.0] * n_joints
        if end_accelerations is None:
            end_accelerations = [0.0] * n_joints
            
        # 시간 배열 생성
        time_array = np.arange(0, duration + dt, dt)
        n_steps = len(time_array)
        
        # 궤적 배열 초기화
        positions = np.zeros((n_steps, n_joints))
        velocities = np.zeros((n_steps, n_joints))
        accelerations = np.zeros((n_steps, n_joints))
        jerks = np.zeros((n_steps, n_joints))
        coefficients = np.zeros((n_joints, 6))  # 각 관절의 6개 계수
        
        # 각 관절에 대해 5차 다항식 계수 계산
        for joint_idx in range(n_joints):
            θ0 = start_angles[joint_idx]
            θf = end_angles[joint_idx]
            ω0 = start_velocities[joint_idx]
            ωf = end_velocities[joint_idx]
            α0 = start_accelerations[joint_idx]
            αf = end_accelerations[joint_idx]
            T = duration
            
            # 5차 다항식의 계수 계산
            # 경계 조건 행렬을 풀어서 계수 구하기
            # [θ0, ω0, α0, θf, ωf, αf] = A * [a0, a1, a2, a3, a4, a5]
            
            A = np.array([
                [1,  0,   0,    0,     0,     0],      # θ(0) = θ0
                [0,  1,   0,    0,     0,     0],      # θ'(0) = ω0
                [0,  0,   2,    0,     0,     0],      # θ''(0) = α0
                [1,  T,   T**2, T**3,  T**4,  T**5],   # θ(T) = θf
                [0,  1,   2*T,  3*T**2, 4*T**3, 5*T**4], # θ'(T) = ωf
                [0,  0,   2,    6*T,   12*T**2, 20*T**3]  # θ''(T) = αf
            ])
            
            b = np.array([θ0, ω0, α0, θf, ωf, αf])
            
            # 계수 계산
            try:
                joint_coeffs = np.linalg.solve(A, b)
                coefficients[joint_idx] = joint_coeffs
            except np.linalg.LinAlgError:
                # 특이행렬인 경우 의사역행렬 사용
                joint_coeffs = np.linalg.pinv(A) @ b
                coefficients[joint_idx] = joint_coeffs
            
            # 시간에 따른 위치, 속도, 가속도, 저크 계산
            a0, a1, a2, a3, a4, a5 = joint_coeffs
            
            for i, t in enumerate(time_array):
                # 위치: θ(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
                positions[i, joint_idx] = (a0 + a1*t + a2*t**2 + 
                                         a3*t**3 + a4*t**4 + a5*t**5)
                
                # 속도: θ'(t) = a1 + 2*a2*t + 3*a3*t² + 4*a4*t³ + 5*a5*t⁴
                velocities[i, joint_idx] = (a1 + 2*a2*t + 3*a3*t**2 + 
                                          4*a4*t**3 + 5*a5*t**4)
                
                # 가속도: θ''(t) = 2*a2 + 6*a3*t + 12*a4*t² + 20*a5*t³
                accelerations[i, joint_idx] = (2*a2 + 6*a3*t + 
                                             12*a4*t**2 + 20*a5*t**3)
                
                # 저크: θ'''(t) = 6*a3 + 24*a4*t + 60*a5*t²
                jerks[i, joint_idx] = 6*a3 + 24*a4*t + 60*a5*t**2
        
        return {
            'time': time_array,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'jerks': jerks,
            'coefficients': coefficients,
            'duration': duration,
            'n_joints': n_joints
        }
    
    def plan_multi_point_trajectory(self, waypoints, durations=None, dt=None):
        """
        다중 점을 거치는 궤적 계획 (여러 구간의 5차 다항식 연결)
        
        Args:
            waypoints (list): 웨이포인트 리스트 [[angles1], [angles2], ...]
            durations (list): 각 구간의 지속 시간
            dt (float): 시간 간격
            
        Returns:
            dict: 전체 궤적 데이터
        """
        n_segments = len(waypoints) - 1
        if n_segments < 1:
            raise ValueError("최소 2개의 웨이포인트가 필요합니다")
        
        if durations is None:
            durations = [self.default_duration] * n_segments
        elif len(durations) != n_segments:
            raise ValueError("구간 수와 지속시간 수가 일치하지 않습니다")
        
        if dt is None:
            dt = self.default_dt
        
        # 전체 궤적 데이터 초기화
        all_time = []
        all_positions = []
        all_velocities = []
        all_accelerations = []
        all_jerks = []
        
        current_time_offset = 0
        
        # 각 구간별로 궤적 계획
        for segment_idx in range(n_segments):
            start_angles = waypoints[segment_idx]
            end_angles = waypoints[segment_idx + 1]
            segment_duration = durations[segment_idx]
            
            # 중간 웨이포인트에서는 속도를 0으로 설정하여 정지
            start_vel = [0.0] * len(start_angles)
            end_vel = [0.0] * len(end_angles)
            
            # 구간 궤적 계획
            segment_traj = self.plan_quintic_trajectory(
                start_angles, end_angles, segment_duration, dt,
                start_vel, end_vel
            )
            
            # 시간 오프셋 적용
            segment_time = segment_traj['time'] + current_time_offset
            
            # 첫 번째 구간이 아니면 시작점 중복 제거
            if segment_idx > 0:
                segment_time = segment_time[1:]
                segment_positions = segment_traj['positions'][1:]
                segment_velocities = segment_traj['velocities'][1:]
                segment_accelerations = segment_traj['accelerations'][1:]
                segment_jerks = segment_traj['jerks'][1:]
            else:
                segment_positions = segment_traj['positions']
                segment_velocities = segment_traj['velocities']
                segment_accelerations = segment_traj['accelerations']
                segment_jerks = segment_traj['jerks']
            
            # 전체 데이터에 추가
            all_time.extend(segment_time)
            all_positions.extend(segment_positions)
            all_velocities.extend(segment_velocities)
            all_accelerations.extend(segment_accelerations)
            all_jerks.extend(segment_jerks)
            
            current_time_offset += segment_duration
        
        return {
            'time': np.array(all_time),
            'positions': np.array(all_positions),
            'velocities': np.array(all_velocities),
            'accelerations': np.array(all_accelerations),
            'jerks': np.array(all_jerks),
            'total_duration': current_time_offset,
            'n_joints': len(waypoints[0]),
            'n_segments': n_segments
        }
    
    def optimize_trajectory_time(self, start_angles, end_angles, 
                               max_velocity=None, max_acceleration=None):
        """
        속도와 가속도 제한을 고려한 최적 궤적 시간 계산
        
        Args:
            start_angles (list): 시작 각도
            end_angles (list): 끝 각도  
            max_velocity (list): 최대 각속도 제한 (라디안/초)
            max_acceleration (list): 최대 각가속도 제한 (라디안/초²)
            
        Returns:
            float: 최적 궤적 시간
        """
        n_joints = len(start_angles)
        
        # 기본 제한값 설정
        if max_velocity is None:
            max_velocity = [np.pi/2] * n_joints  # 90도/초
        if max_acceleration is None:
            max_acceleration = [np.pi] * n_joints  # 180도/초²
        
        min_times = []
        
        for i in range(n_joints):
            angle_diff = abs(end_angles[i] - start_angles[i])
            
            # 속도 제한에 의한 최소 시간
            t_vel = angle_diff / max_velocity[i] if max_velocity[i] > 0 else float('inf')
            
            # 가속도 제한에 의한 최소 시간 (삼각형 프로파일 가정)
            t_acc = np.sqrt(4 * angle_diff / max_acceleration[i]) if max_acceleration[i] > 0 else float('inf')
            
            min_times.append(max(t_vel, t_acc))
        
        # 가장 제한적인 관절의 시간을 전체 시간으로 설정
        optimal_time = max(min_times) * 1.5  # 안전 여유 추가
        
        return max(optimal_time, 1.0)  # 최소 1초
    
    def plot_trajectory_graphs(self, trajectory_data, save_dir=None, filename_prefix="trajectory"):
        """
        궤적 그래프 생성 및 저장
        
        Args:
            trajectory_data (dict): 궤적 데이터
            save_dir (str): 저장 디렉토리
            filename_prefix (str): 파일명 접두사
        """
        time = trajectory_data['time']
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']
        jerks = trajectory_data['jerks']
        n_joints = trajectory_data['n_joints']
        
        # 그래프 설정
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(4, 1, hspace=0.3)
        
        # 1. 각도 그래프
        ax1 = fig.add_subplot(gs[0, 0])
        for joint_idx in range(n_joints):
            ax1.plot(time, np.degrees(positions[:, joint_idx]), 
                    color=self.colors[joint_idx % len(self.colors)],
                    linewidth=2, label=f'Joint {joint_idx + 1}')
        ax1.set_ylabel('각도 (deg)', fontsize=12)
        ax1.set_title('관절 각도 프로파일', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 각속도 그래프
        ax2 = fig.add_subplot(gs[1, 0])
        for joint_idx in range(n_joints):
            ax2.plot(time, np.degrees(velocities[:, joint_idx]), 
                    color=self.colors[joint_idx % len(self.colors)],
                    linewidth=2, label=f'Joint {joint_idx + 1}')
        ax2.set_ylabel('각속도 (deg/s)', fontsize=12)
        ax2.set_title('관절 각속도 프로파일', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 각가속도 그래프
        ax3 = fig.add_subplot(gs[2, 0])
        for joint_idx in range(n_joints):
            ax3.plot(time, np.degrees(accelerations[:, joint_idx]), 
                    color=self.colors[joint_idx % len(self.colors)],
                    linewidth=2, label=f'Joint {joint_idx + 1}')
        ax3.set_ylabel('각가속도 (deg/s²)', fontsize=12)
        ax3.set_title('관절 각가속도 프로파일', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 저크 그래프
        ax4 = fig.add_subplot(gs[3, 0])
        for joint_idx in range(n_joints):
            ax4.plot(time, np.degrees(jerks[:, joint_idx]), 
                    color=self.colors[joint_idx % len(self.colors)],
                    linewidth=2, label=f'Joint {joint_idx + 1}')
        ax4.set_xlabel('시간 (s)', fontsize=12)
        ax4.set_ylabel('저크 (deg/s³)', fontsize=12)
        ax4.set_title('관절 저크 프로파일', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # 파일 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"궤적 그래프 저장됨: {filepath}")
        
        return fig
    
    def plot_3d_trajectory(self, trajectory_data, end_effector_positions, 
                          save_dir=None, filename_prefix="trajectory_3d"):
        """
        3D 공간에서의 End-Effector 궤적 그래프
        
        Args:
            trajectory_data (dict): 궤적 데이터
            end_effector_positions (np.array): End-effector 위치 데이터 [time_steps, 3]
            save_dir (str): 저장 디렉토리
            filename_prefix (str): 파일명 접두사
        """
        fig = plt.figure(figsize=(12, 9))
        
        # 3D 궤적 플롯
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(end_effector_positions[:, 0] * 100,  # m를 cm로 변환
                end_effector_positions[:, 1] * 100,
                end_effector_positions[:, 2] * 100,
                'b-', linewidth=3, alpha=0.8)
        ax1.scatter(end_effector_positions[0, 0] * 100, 
                   end_effector_positions[0, 1] * 100,
                   end_effector_positions[0, 2] * 100,
                   color='green', s=100, label='시작점')
        ax1.scatter(end_effector_positions[-1, 0] * 100, 
                   end_effector_positions[-1, 1] * 100,
                   end_effector_positions[-1, 2] * 100,
                   color='red', s=100, label='끝점')
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('End-Effector 3D 궤적')
        ax1.legend()
        
        # X-Y 평면 투영
        ax2 = fig.add_subplot(222)
        ax2.plot(end_effector_positions[:, 0] * 100, 
                end_effector_positions[:, 1] * 100, 'b-', linewidth=2)
        ax2.scatter(end_effector_positions[0, 0] * 100, 
                   end_effector_positions[0, 1] * 100, 
                   color='green', s=50, label='시작')
        ax2.scatter(end_effector_positions[-1, 0] * 100, 
                   end_effector_positions[-1, 1] * 100, 
                   color='red', s=50, label='끝')
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title('X-Y 평면 투영')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        # X-Z 평면 투영
        ax3 = fig.add_subplot(223)
        ax3.plot(end_effector_positions[:, 0] * 100, 
                end_effector_positions[:, 2] * 100, 'b-', linewidth=2)
        ax3.scatter(end_effector_positions[0, 0] * 100, 
                   end_effector_positions[0, 2] * 100, 
                   color='green', s=50, label='시작')
        ax3.scatter(end_effector_positions[-1, 0] * 100, 
                   end_effector_positions[-1, 2] * 100, 
                   color='red', s=50, label='끝')
        ax3.set_xlabel('X (cm)')
        ax3.set_ylabel('Z (cm)')
        ax3.set_title('X-Z 평면 투영')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Y-Z 평면 투영
        ax4 = fig.add_subplot(224)
        ax4.plot(end_effector_positions[:, 1] * 100, 
                end_effector_positions[:, 2] * 100, 'b-', linewidth=2)
        ax4.scatter(end_effector_positions[0, 1] * 100, 
                   end_effector_positions[0, 2] * 100, 
                   color='green', s=50, label='시작')
        ax4.scatter(end_effector_positions[-1, 1] * 100, 
                   end_effector_positions[-1, 2] * 100, 
                   color='red', s=50, label='끝')
        ax4.set_xlabel('Y (cm)')
        ax4.set_ylabel('Z (cm)')
        ax4.set_title('Y-Z 평면 투영')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # 파일 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"3D 궤적 그래프 저장됨: {filepath}")
        
        return fig
    
    def analyze_trajectory_smoothness(self, trajectory_data):
        """
        궤적의 부드러움 분석
        
        Args:
            trajectory_data (dict): 궤적 데이터
            
        Returns:
            dict: 분석 결과
        """
        jerks = trajectory_data['jerks']
        accelerations = trajectory_data['accelerations']
        velocities = trajectory_data['velocities']
        n_joints = trajectory_data['n_joints']
        
        analysis = {
            'joint_analysis': [],
            'overall_smoothness': 0.0
        }
        
        total_jerk_rms = 0.0
        
        for joint_idx in range(n_joints):
            joint_jerks = jerks[:, joint_idx]
            joint_accels = accelerations[:, joint_idx]
            joint_vels = velocities[:, joint_idx]
            
            # RMS 저크 (부드러움의 척도)
            rms_jerk = np.sqrt(np.mean(joint_jerks**2))
            
            # 최대 절대값들
            max_velocity = np.max(np.abs(joint_vels))
            max_acceleration = np.max(np.abs(joint_accels))
            max_jerk = np.max(np.abs(joint_jerks))
            
            joint_analysis = {
                'joint': joint_idx + 1,
                'rms_jerk': rms_jerk,
                'max_velocity': max_velocity,
                'max_acceleration': max_acceleration,
                'max_jerk': max_jerk,
                'smoothness_score': 1.0 / (1.0 + rms_jerk)  # 저크가 작을수록 높은 점수
            }
            
            analysis['joint_analysis'].append(joint_analysis)
            total_jerk_rms += rms_jerk
        
        # 전체 부드러움 점수
        analysis['overall_smoothness'] = 1.0 / (1.0 + total_jerk_rms / n_joints)
        
        return analysis
    
    def save_trajectory_data(self, trajectory_data, file_path, format='csv'):
        """
        궤적 데이터를 파일로 저장
        
        Args:
            trajectory_data (dict): 궤적 데이터
            file_path (str): 저장 경로
            format (str): 파일 형식 ('csv', 'txt', 'npz')
        """
        time = trajectory_data['time']
        positions = trajectory_data['positions']
        velocities = trajectory_data['velocities']
        accelerations = trajectory_data['accelerations']
        jerks = trajectory_data['jerks']
        n_joints = trajectory_data['n_joints']
        
        if format.lower() == 'csv':
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 헤더 작성
                header = ['Time(s)']
                for i in range(n_joints):
                    header.extend([f'Joint{i+1}_Pos(rad)', f'Joint{i+1}_Vel(rad/s)', 
                                 f'Joint{i+1}_Acc(rad/s2)', f'Joint{i+1}_Jerk(rad/s3)'])
                writer.writerow(header)
                
                # 데이터 작성
                for i, t in enumerate(time):
                    row = [t]
                    for j in range(n_joints):
                        row.extend([positions[i, j], velocities[i, j], 
                                  accelerations[i, j], jerks[i, j]])
                    writer.writerow(row)
                    
        elif format.lower() == 'npz':
            np.savez(file_path, 
                    time=time, positions=positions, velocities=velocities,
                    accelerations=accelerations, jerks=jerks,
                    n_joints=n_joints)
                    
        elif format.lower() == 'txt':
            with open(file_path, 'w', encoding='utf-8') as txtfile:
                txtfile.write(f"Trajectory Data - {n_joints} DOF Robot\n")
                txtfile.write(f"Duration: {trajectory_data.get('duration', 'Unknown')} seconds\n")
                txtfile.write(f"Time steps: {len(time)}\n\n")
                
                txtfile.write("Time(s)\t")
                for i in range(n_joints):
                    txtfile.write(f"Joint{i+1}_Pos(rad)\tJoint{i+1}_Vel(rad/s)\t"
                                f"Joint{i+1}_Acc(rad/s2)\tJoint{i+1}_Jerk(rad/s3)\t")
                txtfile.write("\n")
                
                for i, t in enumerate(time):
                    txtfile.write(f"{t:.6f}\t")
                    for j in range(n_joints):
                        txtfile.write(f"{positions[i, j]:.6f}\t{velocities[i, j]:.6f}\t"
                                    f"{accelerations[i, j]:.6f}\t{jerks[i, j]:.6f}\t")
                    txtfile.write("\n")
        
        print(f"궤적 데이터 저장됨: {file_path}")
    
    def load_trajectory_data(self, file_path, format='csv'):
        """
        파일에서 궤적 데이터 로드
        
        Args:
            file_path (str): 파일 경로
            format (str): 파일 형식
            
        Returns:
            dict: 궤적 데이터
        """
        if format.lower() == 'npz':
            data = np.load(file_path)
            return {
                'time': data['time'],
                'positions': data['positions'],
                'velocities': data['velocities'],
                'accelerations': data['accelerations'],
                'jerks': data['jerks'],
                'n_joints': int(data['n_joints'])
            }
        else:
            raise NotImplementedError(f"Loading format '{format}' not implemented yet")