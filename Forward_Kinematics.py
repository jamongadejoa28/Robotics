import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import koreanize_matplotlib

plt.rcParams['axes.unicode_minus'] = False

class TwoLinkRobot:
    def __init__(self, link_lengths=[1, 1], joint_angles=[0, 0]):
        self.L1 = link_lengths[0]  # 첫 번째 링크 길이
        self.L2 = link_lengths[1]  # 두 번째 링크 길이
        self.theta1 = joint_angles[0]  # 첫 번째 관절 각도 (라디안)
        self.theta2 = joint_angles[1]  # 두 번째 관절 각도 (라디안)
        
        # 로봇의 베이스 위치
        self.base = np.array([0, 0])
        
        # 작업 영역 설정
        self.workspace_radius = self.L1 + self.L2
        
        # 초기 상태 계산
        self.update_kinematics()
        
    def update_kinematics(self):
        """순기구학(Forward Kinematics)을 계산하는 메서드"""
        # 첫 번째 관절 위치 (베이스 위치)
        self.joint1 = self.base
        
        # 두 번째 관절 위치
        self.joint2 = np.array([
            self.joint1[0] + self.L1 * np.cos(self.theta1),
            self.joint1[1] + self.L1 * np.sin(self.theta1)
        ])
        
        # 말단 위치 (end-effector)
        self.end_effector = np.array([
            self.joint2[0] + self.L2 * np.cos(self.theta1 + self.theta2),
            self.joint2[1] + self.L2 * np.sin(self.theta1 + self.theta2)
        ])
        
    def set_joint_angles(self, theta1, theta2):
        """관절 각도 설정 메서드"""       
        self.theta1 = theta1
        self.theta2 = theta2
        self.update_kinematics()
        return True
        
    def set_link_lengths(self, L1, L2):
        """링크 길이 설정 메서드"""
        # 링크 길이 유효성 검사
        if L1 <= 0 or L2 <= 0:
            print("잘못된 링크 길이입니다. 양수 값을 입력해주세요.")
            return False
        
        self.L1 = L1
        self.L2 = L2
        self.workspace_radius = self.L1 + self.L2
        self.update_kinematics()
        return True
    
    def get_robot_points(self):
        """로봇의 각 포인트 위치를 반환"""
        return np.array([self.base, self.joint2, self.end_effector])


def run_simulation():
    """시뮬레이션 실행 함수"""
    # 초기 로봇 설정
    robot = TwoLinkRobot(link_lengths=[2, 1.5], joint_angles=[np.pi/4, np.pi/4])
    
    # 그림 설정 (팝업창으로 표시)
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=0.3)  # 슬라이더를 위한 여백 확보
    
    # 작업 영역 (원) 그리기
    workspace = plt.Circle((0, 0), robot.workspace_radius, fill=False, color='gray', linestyle='--')
    ax.add_patch(workspace)
    
    # 로봇 그리기
    robot_points = robot.get_robot_points()
    link_line, = ax.plot(robot_points[:, 0], robot_points[:, 1], 'o-', linewidth=3)
    
    # 말단 궤적을 표시하기 위한 빈 리스트
    trajectory_x, trajectory_y = [], []
    trajectory_line, = ax.plot([], [], 'r.', markersize=1)
    
    # 축 설정
    max_range = robot.workspace_radius * 1.2
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # 제목과 레이블
    ax.set_title('2DOF 로봇 시뮬레이션', fontsize=16)
    ax.set_xlabel('X 좌표 (m)', fontsize=12)
    ax.set_ylabel('Y 좌표 (m)', fontsize=12)
    
    # 말단 위치 텍스트
    end_effector_text = ax.text(0.05, 0.05, '', transform=ax.transAxes)
    
    # 슬라이더 색상 및 스타일 설정
    slider_color = 'lightgoldenrodyellow'
    
    # 관절 1 각도 슬라이더
    ax_theta1 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=slider_color)
    slider_theta1 = Slider(
        ax=ax_theta1,
        label='관절 1 각도 (도)',
        valmin=-180,
        valmax=180,
        valinit=np.degrees(robot.theta1),
        valfmt='%.1f'
    )
    
    # 관절 2 각도 슬라이더
    ax_theta2 = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=slider_color)
    slider_theta2 = Slider(
        ax=ax_theta2,
        label='관절 2 각도 (도)',
        valmin=-150,
        valmax=150,
        valinit=np.degrees(robot.theta2),
        valfmt='%.1f'
    )
    
    # 링크 1 길이 슬라이더
    ax_L1 = plt.axes([0.2, 0.05, 0.3, 0.03], facecolor=slider_color)
    slider_L1 = Slider(
        ax=ax_L1,
        label='링크 1 길이 (m)',
        valmin=0.5,
        valmax=5.0,
        valinit=robot.L1,
        valfmt='%.1f',
    )
    
    # 링크 2 길이 슬라이더
    ax_L2 = plt.axes([0.2, 0.03, 0.3, 0.03], facecolor=slider_color)
    slider_L2 = Slider(
        ax=ax_L2,
        label='링크 2 길이 (m)',
        valmin=0.5,
        valmax=5.0,
        valinit=robot.L2,
        valfmt='%.1f',
    )
    
    # 리셋 버튼
    ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
    reset_button = Button(ax_reset, '초기화', color=slider_color, hovercolor='0.9')
    
    # 경로 지우기 버튼
    ax_clear = plt.axes([0.65, 0.01, 0.1, 0.03])
    clear_button = Button(ax_clear, '경로 지우기', color=slider_color, hovercolor='0.9')
    
    # 오류 메시지 표시 텍스트
    error_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, 
                         color='red', fontsize=12, ha='center')
    
    def update_plot():
        """로봇 시각화 업데이트 함수"""
        robot_points = robot.get_robot_points()
        link_line.set_data(robot_points[:, 0], robot_points[:, 1])
        
        # 말단 위치 텍스트 업데이트
        end_effector_pos = robot.end_effector
        end_effector_text.set_text(f'말단 위치: ({end_effector_pos[0]:.2f}, {end_effector_pos[1]:.2f})')
        
        # 작업 영역 원 업데이트
        workspace.radius = robot.workspace_radius
        
        # 축 범위 업데이트
        max_range = robot.workspace_radius * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        # 말단 궤적 업데이트
        trajectory_x.append(end_effector_pos[0])
        trajectory_y.append(end_effector_pos[1])
        trajectory_line.set_data(trajectory_x, trajectory_y)
        
        plt.draw()
    
    def update_robot(val=None):
        """슬라이더 값 변경 시 로봇 업데이트 함수"""
        # 라디안 단위로 변환
        theta1_rad = np.radians(slider_theta1.val)
        theta2_rad = np.radians(slider_theta2.val)
        
        # 로봇 각도 설정 및 유효성 검사
        if not robot.set_joint_angles(theta1_rad, theta2_rad):
            error_text.set_text("잘못된 각도 입력: 로봇의 동작 범위를 벗어납니다.")
            return
        error_text.set_text("")
        
        # 로봇 링크 길이 설정
        robot.set_link_lengths(slider_L1.val, slider_L2.val)
        
        update_plot()
    
    def reset_robot(event):
        """로봇 초기화 함수"""
        slider_theta1.reset()
        slider_theta2.reset()
        slider_L1.reset()
        slider_L2.reset()
        error_text.set_text("")
        update_robot()
    
    def clear_trajectory(event):
        """말단 궤적 지우기 함수"""
        trajectory_x.clear()
        trajectory_y.clear()
        trajectory_line.set_data(trajectory_x, trajectory_y)
        plt.draw()
    
    # 이벤트 핸들러 연결
    slider_theta1.on_changed(update_robot)
    slider_theta2.on_changed(update_robot)
    slider_L1.on_changed(update_robot)
    slider_L2.on_changed(update_robot)
    reset_button.on_clicked(reset_robot)
    clear_button.on_clicked(clear_trajectory)
    
    # 초기 플롯 업데이트
    update_plot()
    
    # 그림 표시
    plt.show()

if __name__ == "__main__":
    run_simulation()