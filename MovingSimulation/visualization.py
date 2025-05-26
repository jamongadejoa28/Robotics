"""
visualization.py - MovingSimulation/visualization.py

로봇 시각화 모듈
- 3D 로봇 구조 시각화
- 링크와 관절 표시
- 작업 공간 시각화
- 좌표계 표시
- 애니메이션 지원
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class RobotVisualizer:
    def __init__(self):
        """로봇 시각화기 초기화"""
        # 색상 팔레트 정의
        self.colors = {
            'link': '#4A90E2',      # 링크 색상 (파란색)
            'joint': '#E94B3C',     # 관절 색상 (빨간색)
            'base': '#2D3748',      # 베이스 색상 (진한 회색)
            'end_effector': '#48BB78', # End-effector 색상 (초록색)
            'workspace': '#90CDF4',  # 작업공간 색상 (연한 파란색)
            'trajectory': '#9F7AEA', # 궤적 색상 (보라색)
            'frame_x': '#E53E3E',   # X축 색상 (빨간색)
            'frame_y': '#38A169',   # Y축 색상 (초록색)
            'frame_z': '#3182CE'    # Z축 색상 (파란색)
        }
        
        # 시각화 설정
        self.link_width = 3.0       # 링크 선 두께
        self.joint_size = 8.0       # 관절 점 크기
        self.frame_length = 15.0    # 좌표계 축 길이 (cm)
        self.workspace_alpha = 0.1  # 작업공간 투명도
        
        # 그래프 스타일
        self.grid_alpha = 0.3
        self.background_color = 'white'
        
    def compute_link_positions(self, dh_params, joint_angles):
        """
        DH 파라미터와 관절 각도로부터 각 링크의 위치 계산
        
        Args:
            dh_params (list): DH 파라미터 [[a, alpha, d, theta], ...]
            joint_angles (list): 관절 각도 (라디안)
            
        Returns:
            list: 각 링크 끝의 위치 리스트 [[x, y, z], ...]
        """
        positions = [[0, 0, 0]]  # 베이스 위치
        T_cumulative = np.eye(4)
        
        for i, (dh_param, joint_angle) in enumerate(zip(dh_params, joint_angles)):
            a, alpha_deg, d, theta_offset_deg = dh_param
            
            # 각도를 라디안으로 변환
            alpha = np.radians(alpha_deg)
            theta_offset = np.radians(theta_offset_deg)
            theta = joint_angle + theta_offset
            
            # cm를 m로 변환
            a_m = a / 100.0
            d_m = d / 100.0
            
            # DH 변환 행렬
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            
            T_i = np.array([
                [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a_m * cos_theta],
                [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a_m * sin_theta],
                [0,          sin_alpha,              cos_alpha,             d_m],
                [0,          0,                      0,                     1]
            ])
            
            T_cumulative = np.dot(T_cumulative, T_i)
            
            # 현재 링크 끝의 위치 추출 (m를 cm로 변환)
            position = T_cumulative[:3, 3] * 100  # cm 단위로 변환
            positions.append(position.tolist())
        
        return positions
    
    def draw_robot_links(self, ax, link_positions):
        """
        로봇 링크를 3D 공간에 그리기
        
        Args:
            ax: matplotlib 3D axis
            link_positions (list): 링크 위치 리스트
        """
        # 링크들을 선으로 연결
        for i in range(len(link_positions) - 1):
            start_pos = link_positions[i]
            end_pos = link_positions[i + 1]
            
            # 링크 색상 (베이스는 다른 색상)
            if i == 0:
                color = self.colors['base']
                linewidth = self.link_width + 1
                alpha = 1.0
            else:
                color = self.colors['link']
                linewidth = self.link_width
                alpha = 0.8
            
            ax.plot([start_pos[0], end_pos[0]], 
                   [start_pos[1], end_pos[1]], 
                   [start_pos[2], end_pos[2]], 
                   color=color, linewidth=linewidth, alpha=alpha)
    
    def draw_joints(self, ax, link_positions):
        """
        관절을 구 형태로 그리기
        
        Args:
            ax: matplotlib 3D axis
            link_positions (list): 링크 위치 리스트
        """
        for i, pos in enumerate(link_positions):
            if i == 0:
                # 베이스
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=self.colors['base'], s=self.joint_size * 2, 
                          alpha=1.0, marker='s', edgecolors='black', linewidth=1)
            elif i == len(link_positions) - 1:
                # End-effector
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=self.colors['end_effector'], s=self.joint_size * 1.5, 
                          alpha=1.0, marker='^', edgecolors='black', linewidth=1)
            else:
                # 일반 관절
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=self.colors['joint'], s=self.joint_size, 
                          alpha=1.0, marker='o', edgecolors='black', linewidth=1)
    
    def draw_coordinate_frames(self, ax, transformation_matrices, frame_length=None):
        """
        각 링크의 좌표계 그리기
        
        Args:
            ax: matplotlib 3D axis
            transformation_matrices (list): 각 링크의 변환 행렬 리스트
            frame_length (float): 좌표계 축 길이 (cm)
        """
        if frame_length is None:
            frame_length = self.frame_length
        
        for i, T in enumerate(transformation_matrices):
            # 원점 위치 (m를 cm로 변환)
            origin = T[:3, 3] * 100
            
            # 각 축의 방향 벡터
            x_axis = T[:3, 0] * frame_length / 100  # cm를 m로 변환 후 스케일링
            y_axis = T[:3, 1] * frame_length / 100
            z_axis = T[:3, 2] * frame_length / 100
            
            # X축 (빨간색)
            ax.quiver(origin[0], origin[1], origin[2],
                     x_axis[0] * 100, x_axis[1] * 100, x_axis[2] * 100,
                     color=self.colors['frame_x'], alpha=0.8, 
                     arrow_length_ratio=0.1, linewidth=2)
            
            # Y축 (초록색)
            ax.quiver(origin[0], origin[1], origin[2],
                     y_axis[0] * 100, y_axis[1] * 100, y_axis[2] * 100,
                     color=self.colors['frame_y'], alpha=0.8, 
                     arrow_length_ratio=0.1, linewidth=2)
            
            # Z축 (파란색)
            ax.quiver(origin[0], origin[1], origin[2],
                     z_axis[0] * 100, z_axis[1] * 100, z_axis[2] * 100,
                     color=self.colors['frame_z'], alpha=0.8, 
                     arrow_length_ratio=0.1, linewidth=2)
            
            # 프레임 번호 표시
            if i > 0:  # 베이스 프레임 제외
                ax.text(origin[0] + 5, origin[1] + 5, origin[2] + 5, 
                       f'{i}', fontsize=8, color='black', alpha=0.7)
    
    def draw_workspace(self, ax, dof, workspace_points=None):
        """
        로봇의 작업 공간 시각화
        
        Args:
            ax: matplotlib 3D axis
            dof (int): 자유도
            workspace_points (np.array): 작업공간 점들 (선택적)
        """
        if workspace_points is not None:
            # 제공된 작업공간 점들 사용
            points = workspace_points * 100  # m를 cm로 변환
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=self.colors['workspace'], alpha=self.workspace_alpha,
                      s=1, marker='.')
        else:
            # 간단한 작업공간 추정 (원형 또는 구형)
            self._draw_simple_workspace(ax, dof)
    
    def _draw_simple_workspace(self, ax, dof):
        """
        간단한 작업공간 표시 (추정값)
        
        Args:
            ax: matplotlib 3D axis
            dof (int): 자유도
        """
        # 기본 작업반경 추정 (cm)
        if dof == 1:
            max_reach = 50
        elif dof == 2:
            max_reach = 80
        else:
            max_reach = 100
        
        # 구형 작업공간 경계 표시
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        x_sphere = max_reach * np.outer(np.cos(u), np.sin(v))
        y_sphere = max_reach * np.outer(np.sin(u), np.sin(v))
        z_sphere = max_reach * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                         color=self.colors['workspace'], 
                         alpha=self.workspace_alpha, linewidth=0.5)
    
    def draw_trajectory(self, ax, trajectory_points, color=None, label="Trajectory"):
        """
        End-effector 궤적 그리기
        
        Args:
            ax: matplotlib 3D axis
            trajectory_points (np.array): 궤적 점들 [n_points, 3] (m 단위)
            color (str): 궤적 색상
            label (str): 궤적 라벨
        """
        if color is None:
            color = self.colors['trajectory']
        
        # m를 cm로 변환
        points = trajectory_points * 100
        
        # 궤적 선 그리기
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 
               color=color, linewidth=2, alpha=0.8, label=label)
        
        # 시작점과 끝점 표시
        ax.scatter(points[0, 0], points[0, 1], points[0, 2], 
                  c='green', s=50, marker='o', alpha=1.0, label='Start')
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
                  c='red', s=50, marker='s', alpha=1.0, label='End')
    
    def setup_3d_plot(self, ax, title="Robot Visualization", xlim=(-100, 100), 
                     ylim=(-100, 100), zlim=(-100, 100)):
        """
        3D 플롯 기본 설정
        
        Args:
            ax: matplotlib 3D axis
            title (str): 그래프 제목
            xlim (tuple): X축 범위
            ylim (tuple): Y축 범위
            zlim (tuple): Z축 범위
        """
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        ax.set_xlabel('X (cm)', fontsize=12)
        ax.set_ylabel('Y (cm)', fontsize=12)
        ax.set_zlabel('Z (cm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 격자 표시
        ax.grid(True, alpha=self.grid_alpha)
        
        # 배경색 설정
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # 축의 색상 설정
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # 동일한 스케일 설정
        max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]) / 2
        mid_x = (xlim[0] + xlim[1]) / 2
        mid_y = (ylim[0] + ylim[1]) / 2
        mid_z = (zlim[0] + zlim[1]) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    def create_robot_schematic(self, dh_params, joint_angles, save_path=None):
        """
        로봇의 2D 개략도 생성 (측면도 및 평면도)
        
        Args:
            dh_params (list): DH 파라미터
            joint_angles (list): 관절 각도
            save_path (str): 저장 경로 (선택적)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 링크 위치 계산
        link_positions = self.compute_link_positions(dh_params, joint_angles)
        positions = np.array(link_positions)
        
        # 측면도 (X-Z 평면)
        ax1.plot(positions[:, 0], positions[:, 2], 'o-', 
                color=self.colors['link'], linewidth=3, markersize=8)
        ax1.scatter(positions[0, 0], positions[0, 2], 
                   c=self.colors['base'], s=100, marker='s', 
                   edgecolors='black', linewidth=2, label='Base')
        ax1.scatter(positions[-1, 0], positions[-1, 2], 
                   c=self.colors['end_effector'], s=100, marker='^', 
                   edgecolors='black', linewidth=2, label='End-Effector')
        
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Z (cm)')
        ax1.set_title('측면도 (X-Z Plane)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # 평면도 (X-Y 평면)
        ax2.plot(positions[:, 0], positions[:, 1], 'o-', 
                color=self.colors['link'], linewidth=3, markersize=8)
        ax2.scatter(positions[0, 0], positions[0, 1], 
                   c=self.colors['base'], s=100, marker='s', 
                   edgecolors='black', linewidth=2, label='Base')
        ax2.scatter(positions[-1, 0], positions[-1, 1], 
                   c=self.colors['end_effector'], s=100, marker='^', 
                   edgecolors='black', linewidth=2, label='End-Effector')
        
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title('평면도 (X-Y Plane)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"로봇 개략도 저장됨: {save_path}")
        
        return fig
    
    def animate_robot_motion(self, ax, dh_params, trajectory_data, interval=100):
        """
        로봇 움직임 애니메이션 설정
        
        Args:
            ax: matplotlib 3D axis
            dh_params (list): DH 파라미터
            trajectory_data (dict): 궤적 데이터
            interval (int): 애니메이션 간격 (ms)
        """
        from matplotlib.animation import FuncAnimation
        
        positions_history = trajectory_data['positions']
        n_frames = len(positions_history)
        
        def animate_frame(frame):
            ax.clear()
            self.setup_3d_plot(ax, f"Robot Animation - Frame {frame+1}/{n_frames}")
            
            # 현재 프레임의 관절 각도
            current_angles = positions_history[frame]
            
            # 로봇 그리기
            link_positions = self.compute_link_positions(dh_params, current_angles)
            self.draw_robot_links(ax, link_positions)
            self.draw_joints(ax, link_positions)
            
            # 궤적 히스토리 표시 (현재까지의 경로)
            if frame > 0:
                history_positions = []
                for i in range(frame + 1):
                    hist_angles = positions_history[i]
                    hist_link_pos = self.compute_link_positions(dh_params, hist_angles)
                    history_positions.append(hist_link_pos[-1])  # End-effector 위치만
                
                history_array = np.array(history_positions)
                ax.plot(history_array[:, 0], history_array[:, 1], history_array[:, 2],
                       color=self.colors['trajectory'], alpha=0.5, linewidth=1)
        
        animation = FuncAnimation(fig=ax.figure, func=animate_frame, 
                                frames=n_frames, interval=interval, repeat=True)
        
        return animation
    
    def create_joint_limit_visualization(self, dh_params, joint_limits, save_path=None):
        """
        관절 제한 범위 시각화
        
        Args:
            dh_params (list): DH 파라미터
            joint_limits (dict): 관절 제한 {joint_idx: (min_deg, max_deg)}
            save_path (str): 저장 경로
        """
        n_joints = len(dh_params)
        fig, axes = plt.subplots(2, (n_joints + 1) // 2, figsize=(4 * n_joints, 8))
        if n_joints == 1:
            axes = [axes]
        elif n_joints <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()[:n_joints]
        
        for joint_idx in range(n_joints):
            ax = axes[joint_idx]
            
            if joint_idx in joint_limits:
                min_angle, max_angle = joint_limits[joint_idx]
                
                # 원형 차트로 관절 범위 표시
                angles = np.linspace(np.radians(min_angle), np.radians(max_angle), 100)
                
                # 허용 범위
                ax.fill_between(angles, 0, 1, alpha=0.3, color='green', label='허용 범위')
                
                # 금지 구역
                if min_angle > -180:
                    forbidden_angles1 = np.linspace(-np.pi, np.radians(min_angle), 50)
                    ax.fill_between(forbidden_angles1, 0, 1, alpha=0.3, color='red', label='금지 구역')
                
                if max_angle < 180:
                    forbidden_angles2 = np.linspace(np.radians(max_angle), np.pi, 50)
                    ax.fill_between(forbidden_angles2, 0, 1, alpha=0.3, color='red')
                
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(0, 1.2)
                ax.set_xlabel('각도 (라디안)')
                ax.set_title(f'관절 {joint_idx + 1} 제한 범위\n[{min_angle}°, {max_angle}°]')
                ax.legend()
                
                # X축 라벨을 도 단위로 표시
                ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                ax.set_xticklabels(['-180°', '-90°', '0°', '90°', '180°'])
            else:
                ax.text(0.5, 0.5, f'관절 {joint_idx + 1}\n제한 없음', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        # 사용하지 않는 서브플롯 숨기기
        for i in range(n_joints, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"관절 제한 시각화 저장됨: {save_path}")
        
        return fig
    
    def create_workspace_analysis(self, workspace_points, save_path=None):
        """
        작업공간 분석 시각화
        
        Args:
            workspace_points (np.array): 작업공간 점들 (m 단위)
            save_path (str): 저장 경로
        """
        fig = plt.figure(figsize=(15, 10))
        
        # m를 cm로 변환
        points = workspace_points * 100
        
        # 3D 작업공간
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=points[:, 2], cmap='viridis', alpha=0.6, s=1)
        ax1.set_xlabel('X (cm)')
        ax1.set_ylabel('Y (cm)')
        ax1.set_zlabel('Z (cm)')
        ax1.set_title('3D 작업공간')
        
        # X-Y 평면 투영
        ax2 = fig.add_subplot(222)
        scatter = ax2.scatter(points[:, 0], points[:, 1], 
                            c=points[:, 2], cmap='viridis', alpha=0.6, s=1)
        ax2.set_xlabel('X (cm)')
        ax2.set_ylabel('Y (cm)')
        ax2.set_title('X-Y 평면 투영')
        ax2.axis('equal')
        plt.colorbar(scatter, ax=ax2, label='Z (cm)')
        
        # 작업공간 범위 분석
        ax3 = fig.add_subplot(223)
        ranges = {
            'X': [points[:, 0].min(), points[:, 0].max()],
            'Y': [points[:, 1].min(), points[:, 1].max()],
            'Z': [points[:, 2].min(), points[:, 2].max()]
        }
        
        axes_names = list(ranges.keys())
        min_vals = [ranges[axis][0] for axis in axes_names]
        max_vals = [ranges[axis][1] for axis in axes_names]
        
        x_pos = np.arange(len(axes_names))
        width = 0.35
        
        ax3.bar(x_pos - width/2, min_vals, width, label='최소값', alpha=0.7)
        ax3.bar(x_pos + width/2, max_vals, width, label='최대값', alpha=0.7)
        
        ax3.set_xlabel('축')
        ax3.set_ylabel('거리 (cm)')
        ax3.set_title('작업공간 범위')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(axes_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 도달거리 히스토그램
        ax4 = fig.add_subplot(224)
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        ax4.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('원점으로부터의 거리 (cm)')
        ax4.set_ylabel('빈도')
        ax4.set_title('도달거리 분포')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"작업공간 분석 저장됨: {save_path}")
        
        return fig
    
    def set_color_scheme(self, scheme='default'):
        """
        색상 스키마 변경
        
        Args:
            scheme (str): 색상 스키마 ('default', 'dark', 'colorful', 'minimal')
        """
        if scheme == 'dark':
            self.colors.update({
                'link': '#64B5F6',
                'joint': '#FF7043',
                'base': '#37474F',
                'end_effector': '#66BB6A',
                'workspace': '#42A5F5',
                'trajectory': '#AB47BC'
            })
        elif scheme == 'colorful':
            self.colors.update({
                'link': '#FF6B6B',
                'joint': '#4ECDC4',
                'base': '#45B7D1',
                'end_effector': '#F7DC6F',
                'workspace': '#BB8FCE',
                'trajectory': '#85C1E9'
            })
        elif scheme == 'minimal':
            self.colors.update({
                'link': '#333333',
                'joint': '#666666',
                'base': '#000000',
                'end_effector': '#999999',
                'workspace': '#CCCCCC',
                'trajectory': '#555555'
            })
        # 'default'는 이미 초기화에서 설정됨