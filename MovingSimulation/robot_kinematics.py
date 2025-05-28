"""
robot_kinematics.py - MovingSimulation/robot_kinematics.py

로봇 운동학 계산을 담당하는 핵심 모듈
- Forward Kinematics (정기구학)
- Inverse Kinematics (역기구학)
- DH 변환 행렬 계산
- 자코비언 행렬 계산
- 특이점 검출 및 회피
"""

import numpy as np
from scipy.optimize import fsolve, minimize, least_squares
import warnings

class RobotKinematics:
    def __init__(self):
        """로봇 운동학 클래스 초기화"""
        # 관절 제한값 설정 (도 단위)
        self.joint_limits = {
            0: (-180, 180),
            1: (-135, 135),
            2: (-90, 90),
            3: (-180, 180),
            4: (-120, 120),
            5: (-180, 180),
        }
        
        # 특이점 임계값
        self.singularity_threshold = 1e-6
        self.condition_number_threshold = 100
        
    def dh_transform_matrix(self, a, alpha, d, theta):
        """DH 파라미터로부터 동차 변환 행렬 계산
        
        Args:
            a (float): 링크 길이 (cm)
            alpha (float): 링크 트위스트 (라디안)
            d (float): 링크 오프셋 (cm)  
            theta (float): 관절 각도 (라디안)
        """
        # cm를 m로 변환
        a_m = a / 100.0
        d_m = d / 100.0
        
        # 삼각함수 값 계산
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # DH 변환 행렬 구성
        T = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a_m * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a_m * sin_theta],
            [0,          sin_alpha,              cos_alpha,             d_m],
            [0,          0,                      0,                     1]
        ])
        
        return T
    
    def forward_kinematics(self, dh_params, joint_angles):
        """정기구학 계산 - 관절 각도로부터 end-effector 위치 계산
        
        Args:
            dh_params (list): DH 파라미터 리스트 [[a, alpha, d, theta], ...]
            joint_angles (list): 관절 각도 리스트 (라디안)
            
        Returns:
            numpy.ndarray: End-effector의 4x4 동차 변환 행렬
        """
        # 단위 행렬로 시작
        T_total = np.eye(4)
        
        # 각 링크에 대해 변환 행렬 계산 및 곱셈
        for i, (dh_param, joint_angle) in enumerate(zip(dh_params, joint_angles)):
            a, alpha_deg, d, theta_offset_deg = dh_param
            
            # 각도를 라디안으로 변환
            alpha = np.radians(alpha_deg)
            theta_offset = np.radians(theta_offset_deg)
            
            # 실제 관절 각도 = 오프셋 + 관절 변수
            theta = joint_angle + theta_offset
            
            # DH 변환 행렬 계산 (모든 각도를 라디안으로 전달)
            T_i = self.dh_transform_matrix(a, alpha, d, theta)
            
            # 누적 변환
            T_total = np.dot(T_total, T_i)
        
        return T_total
    
    def compute_jacobian(self, dh_params, joint_angles):
        """자코비언 행렬 계산
        
        Args:
            dh_params (list): DH 파라미터 리스트
            joint_angles (list): 현재 관절 각도 (라디안)
            
        Returns:
            numpy.ndarray: 6xN 자코비언 행렬 (N은 DOF)
        """
        n_joints = len(joint_angles)
        jacobian = np.zeros((6, n_joints))
        
        # 각 관절의 원점과 z축 방향 벡터 계산
        origins = [np.array([0, 0, 0])]
        z_axes = [np.array([0, 0, 1])]
        
        T_cumulative = np.eye(4)
        
        # 각 관절까지의 변환 행렬 계산
        for i in range(n_joints):
            a, alpha_deg, d, theta_offset_deg = dh_params[i]
            alpha = np.radians(alpha_deg)
            theta_offset = np.radians(theta_offset_deg)
            theta = joint_angles[i] + theta_offset
            
            T_i = self.dh_transform_matrix(a, alpha, d, theta)
            T_cumulative = np.dot(T_cumulative, T_i)
            
            # 관절 i의 원점과 z축
            origin_i = T_cumulative[:3, 3]
            z_axis_i = T_cumulative[:3, 2]
            
            origins.append(origin_i)
            z_axes.append(z_axis_i)
        
        # End-effector 위치
        end_effector_pos = origins[-1]
        
        # 각 관절에 대한 자코비언 열 계산
        for i in range(n_joints):
            # 병진 속도 부분: z_i × (p_e - p_i)
            p_diff = end_effector_pos - origins[i]
            v_linear = np.cross(z_axes[i], p_diff)
            
            # 각속도 부분: z_i
            v_angular = z_axes[i]
            
            # 자코비언 열 구성
            jacobian[:3, i] = v_linear
            jacobian[3:, i] = v_angular
        
        return jacobian
    
    def check_singularity(self, jacobian):
        """특이점 검사"""
        # 자코비언이 정사각행렬이 아닌 경우 처리
        if jacobian.shape[0] != jacobian.shape[1]:
            jacobian_square = np.dot(jacobian.T, jacobian)
        else:
            jacobian_square = jacobian
        
        # 행렬식 계산
        try:
            det = np.linalg.det(jacobian_square)
        except:
            det = 0.0
        
        # 조건수 계산
        try:
            cond_num = np.linalg.cond(jacobian_square)
        except:
            cond_num = float('inf')
        
        # 계수 계산
        rank = np.linalg.matrix_rank(jacobian)
        
        # 특이점 판정
        is_singular = (abs(det) < self.singularity_threshold or 
                      cond_num > self.condition_number_threshold)
        
        return {
            'is_singular': is_singular,
            'determinant': det,
            'condition_number': cond_num,
            'rank': rank
        }
    
    def avoid_singularity(self, jacobian, joint_angles, damping_factor=0.1):
        """감쇠 최소제곱법을 이용한 특이점 회피"""
        # 감쇠 행렬 추가
        n = jacobian.shape[1]
        damping_matrix = damping_factor * np.eye(n)
        
        # 감쇠 최소제곱 의사역행렬 계산
        try:
            JTJ_damped = np.dot(jacobian.T, jacobian) + damping_matrix
            J_pinv = np.dot(np.linalg.inv(JTJ_damped), jacobian.T)
        except:
            J_pinv = np.linalg.pinv(jacobian)
        
        return J_pinv
    
    def inverse_kinematics(self, dh_params, target_position, target_orientation=None, 
                          initial_guess=None, method='numerical'):
        """역기구학 계산"""
        n_joints = len(dh_params)
        
        # 초기 추정값 설정
        if initial_guess is None:
            initial_guess = [0.0] * n_joints
        
        # 목표 동차 변환 행렬 구성
        target_T = self.create_target_transform_matrix(target_position, target_orientation)
        
        if method == 'numerical':
            return self._inverse_kinematics_numerical_improved(dh_params, target_T, initial_guess)
        elif method == 'analytical':
            return self._inverse_kinematics_analytical(dh_params, target_position, target_orientation)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def create_target_transform_matrix(self, position, orientation=None):
        """목표 위치와 자세로부터 동차 변환 행렬 생성"""
        T = np.eye(4)
        T[:3, 3] = position
        
        if orientation is not None:
            roll, pitch, yaw = orientation
            
            # 회전 행렬 계산 (ZYX Euler angles)
            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw),  np.cos(yaw), 0],
                           [0,            0,           1]])
            
            R_y = np.array([[np.cos(pitch),  0, np.sin(pitch)],
                           [0,              1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])
            
            R_x = np.array([[1, 0,             0],
                           [0, np.cos(roll), -np.sin(roll)],
                           [0, np.sin(roll),  np.cos(roll)]])
            
            R = np.dot(R_z, np.dot(R_y, R_x))
            T[:3, :3] = R
        
        return T
    
    def _inverse_kinematics_numerical_improved(self, dh_params, target_T, initial_guess):
        """개선된 수치해석적 역기구학 해법"""
        n_joints = len(dh_params)
        
        def objective_function(joint_angles):
            """목적 함수 - 위치와 자세 오차의 제곱합"""
            try:
                current_T = self.forward_kinematics(dh_params, joint_angles)
                
                # 위치 오차
                pos_error = target_T[:3, 3] - current_T[:3, 3]
                
                # 자세 오차 (회전 행렬 차이)
                if target_T.shape == (4, 4) and n_joints >= 3:
                    R_target = target_T[:3, :3]
                    R_current = current_T[:3, :3]
                    R_error = np.dot(R_target, R_current.T)
                    
                    # 회전 행렬을 축-각 표현으로 변환하여 오차 계산
                    trace_R = np.trace(R_error)
                    trace_R = np.clip(trace_R, -1, 3)
                    
                    if trace_R >= 3:
                        angle_error = 0.0
                    else:
                        angle_error = np.arccos((trace_R - 1) / 2)
                    
                    # 회전축 계산
                    if abs(angle_error) > 1e-6:
                        axis_error = np.array([
                            R_error[2, 1] - R_error[1, 2],
                            R_error[0, 2] - R_error[2, 0], 
                            R_error[1, 0] - R_error[0, 1]
                        ]) / (2 * np.sin(angle_error))
                        rot_error = axis_error * angle_error
                    else:
                        rot_error = np.zeros(3)
                else:
                    rot_error = np.zeros(3)
                
                # 전체 오차 벡터
                error_vector = np.concatenate([pos_error * 10, rot_error])
                return np.sum(error_vector**2)
                
            except Exception as e:
                return 1e6
        
        def residual_function(joint_angles):
            """잔차 함수 - least_squares용"""
            try:
                current_T = self.forward_kinematics(dh_params, joint_angles)
                
                # 위치 오차
                pos_error = target_T[:3, 3] - current_T[:3, 3]
                
                # 자세 오차
                if n_joints >= 3 and target_T.shape == (4, 4):
                    z_target = target_T[:3, 2]
                    z_current = current_T[:3, 2]
                    z_error = z_target - z_current
                    
                    if n_joints >= 6:
                        x_target = target_T[:3, 0]
                        x_current = current_T[:3, 0]
                        x_error = x_target - x_current
                        return np.concatenate([pos_error, z_error, x_error])
                    else:
                        return np.concatenate([pos_error, z_error])
                else:
                    if n_joints == 1:
                        return [pos_error[0]]
                    elif n_joints == 2:
                        return pos_error[:2]
                    else:
                        return pos_error
                        
            except Exception as e:
                if n_joints == 1:
                    return [1000.0]
                elif n_joints == 2:
                    return [1000.0, 1000.0]
                elif n_joints == 3:
                    return [1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0]
                else:
                    return [1000.0] * (3 + min(n_joints - 3, 3))
        
        try:
            # 관절 제한 설정
            bounds = []
            for i in range(len(initial_guess)):
                if i in self.joint_limits:
                    min_angle, max_angle = self.joint_limits[i]
                    bounds.append((np.radians(min_angle), np.radians(max_angle)))
                else:
                    bounds.append((-np.pi, np.pi))
            
            # Method 1: scipy.optimize.minimize 사용
            result1 = minimize(objective_function, initial_guess, 
                             method='L-BFGS-B', bounds=bounds)
            
            if result1.success and result1.fun < 1e-4:
                return result1.x.tolist()
            
            # Method 2: scipy.optimize.least_squares 사용
            result2 = least_squares(residual_function, initial_guess, 
                                  bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
            
            if result2.success and result2.cost < 1e-4:
                return result2.x.tolist()
            
            # Method 3: fsolve 시도
            if n_joints <= 3:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        solution = fsolve(self._ik_equations_improved, initial_guess, 
                                        args=(dh_params, target_T))
                    
                    if self._verify_ik_solution(solution, dh_params, target_T):
                        return solution.tolist()
                except:
                    pass
            
            return None
        
        except Exception as e:
            print(f"Numerical IK error: {e}")
            return None
    
    def _ik_equations_improved(self, joint_angles, dh_params, target_T):
        """개선된 역기구학 방정식 (fsolve용)"""
        try:
            n_joints = len(joint_angles)
            current_T = self.forward_kinematics(dh_params, joint_angles)
            
            # 위치 오차
            pos_error = target_T[:3, 3] - current_T[:3, 3]
            
            if n_joints == 1:
                return [pos_error[0]]
            elif n_joints == 2:
                return pos_error[:2].tolist()
            elif n_joints == 3:
                return pos_error.tolist()
            elif n_joints == 4:
                z_target = target_T[:3, 2]
                z_current = current_T[:3, 2]
                z_error = np.dot(z_target, z_current) - 1
                return np.concatenate([pos_error, [z_error]]).tolist()
            elif n_joints == 5:
                z_target = target_T[:3, 2]
                z_current = current_T[:3, 2]
                z_error = np.dot(z_target, z_current) - 1
                
                y_target = target_T[:3, 1]
                y_current = current_T[:3, 1]
                y_error = np.dot(y_target, y_current) - 1
                
                return np.concatenate([pos_error, [z_error, y_error]]).tolist()
            elif n_joints >= 6:
                # 회전 행렬 오차를 축-각 표현으로 변환
                R_target = target_T[:3, :3]
                R_current = current_T[:3, :3]
                R_error = np.dot(R_target, R_current.T)
                
                trace_R = np.trace(R_error)
                trace_R = np.clip(trace_R, -1, 3)
                
                if trace_R >= 2.99:
                    rot_error = np.zeros(3)
                else:
                    angle = np.arccos((trace_R - 1) / 2)
                    if abs(angle) < 1e-6:
                        rot_error = np.zeros(3)
                    else:
                        axis = np.array([
                            R_error[2, 1] - R_error[1, 2],
                            R_error[0, 2] - R_error[2, 0],
                            R_error[1, 0] - R_error[0, 1]
                        ]) / (2 * np.sin(angle))
                        rot_error = axis * angle
                
                return np.concatenate([pos_error, rot_error]).tolist()
            else:
                return pos_error.tolist()
                
        except Exception as e:
            n_joints = len(joint_angles)
            if n_joints == 1:
                return [1000.0]
            elif n_joints == 2:
                return [1000.0, 1000.0]
            elif n_joints == 3:
                return [1000.0, 1000.0, 1000.0]
            elif n_joints == 4:
                return [1000.0, 1000.0, 1000.0, 1000.0]
            elif n_joints == 5:
                return [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
            else:
                return [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    
    def _verify_ik_solution(self, solution, dh_params, target_T, tolerance=1e-3):
        """역기구학 해의 검증"""
        try:
            current_T = self.forward_kinematics(dh_params, solution)
            pos_error = np.linalg.norm(target_T[:3, 3] - current_T[:3, 3])
            return pos_error < tolerance
        except:
            return False
    
    def _inverse_kinematics_analytical(self, dh_params, target_position, target_orientation):
        """해석적 역기구학 해법 (특정 로봇 구조에 대해)"""
        n_joints = len(dh_params)
        
        if n_joints == 2:
            return self._2dof_planar_ik(dh_params, target_position)
        elif n_joints == 3:
            return self._3dof_planar_ik(dh_params, target_position)
        else:
            return self._inverse_kinematics_numerical_improved(dh_params, 
                self.create_target_transform_matrix(target_position, target_orientation), 
                [0.0] * n_joints)
    
    def _2dof_planar_ik(self, dh_params, target_position):
        """2DOF 평면 로봇의 해석적 역기구학"""
        x, y = target_position[0], target_position[1]
        
        # 링크 길이 (cm를 m로 변환)
        l1 = dh_params[0][0] / 100.0
        l2 = dh_params[1][0] / 100.0
        
        # 목표점까지의 거리
        r = np.sqrt(x**2 + y**2)
        
        # 도달 가능성 검사
        if r > (l1 + l2) or r < abs(l1 - l2):
            return None
        
        # 코사인 법칙으로 θ2 계산
        cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        # 수치 오차 처리
        if abs(cos_theta2) > 1:
            cos_theta2 = np.sign(cos_theta2)
        
        # 두 가지 해 (팔꿈치 위/아래)
        theta2_1 = np.arccos(cos_theta2)
        theta2_2 = -theta2_1
        
        # θ1 계산
        for theta2 in [theta2_1, theta2_2]:
            denominator = l1 + l2 * np.cos(theta2)
            if abs(denominator) > 1e-6:
                theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), denominator)
                
                if self._check_joint_limits([theta1, theta2]):
                    return [theta1, theta2]
        
        return None
    
    def _3dof_planar_ik(self, dh_params, target_position):
        """3DOF 평면 로봇의 해석적 역기구학"""
        if len(dh_params) >= 2:
            two_dof_solution = self._2dof_planar_ik(dh_params[:2], target_position)
            if two_dof_solution:
                return two_dof_solution + [0.0]
        
        return None
    
    def _check_joint_limits(self, joint_angles):
        """관절 제한 검사"""
        for i, angle in enumerate(joint_angles):
            if i in self.joint_limits:
                min_limit, max_limit = self.joint_limits[i]
                angle_deg = np.degrees(angle)
                if angle_deg < min_limit or angle_deg > max_limit:
                    return False
        return True
    
    def get_joint_limits(self, joint_index):
        """특정 관절의 제한값 반환"""
        return self.joint_limits.get(joint_index, (-180, 180))
    
    def compute_workspace(self, dh_params, resolution=50):
        """로봇의 작업 공간 계산"""
        n_joints = len(dh_params)
        workspace_points = []
        
        # 각 관절의 각도 범위를 균등 분할
        angle_ranges = []
        for i in range(n_joints):
            if i in self.joint_limits:
                min_angle, max_angle = self.joint_limits[i]
            else:
                min_angle, max_angle = -180, 180
            
            angles = np.linspace(np.radians(min_angle), np.radians(max_angle), resolution)
            angle_ranges.append(angles)
        
        # 모든 관절 각도 조합에 대해 end-effector 위치 계산
        if n_joints == 1:
            for angle1 in angle_ranges[0]:
                T = self.forward_kinematics(dh_params, [angle1])
                workspace_points.append(T[:3, 3])
        elif n_joints == 2:
            for angle1 in angle_ranges[0]:
                for angle2 in angle_ranges[1]:
                    T = self.forward_kinematics(dh_params, [angle1, angle2])
                    workspace_points.append(T[:3, 3])
        elif n_joints >= 3:
            # 3DOF 이상은 샘플링 수를 줄여서 계산
            reduced_resolution = min(resolution, 20)
            for i in range(n_joints):
                angle_ranges[i] = np.linspace(np.radians(-90), np.radians(90), reduced_resolution)
            
            # 랜덤 샘플링으로 작업공간 추정
            for _ in range(1000):
                joint_angles = []
                for i in range(n_joints):
                    angle = np.random.choice(angle_ranges[i])
                    joint_angles.append(angle)
                
                T = self.forward_kinematics(dh_params, joint_angles)
                workspace_points.append(T[:3, 3])
        
        return np.array(workspace_points)
    
    def compute_manipulability(self, jacobian):
        """조작성 지수 계산"""
        try:
            if jacobian.shape[0] != jacobian.shape[1]:
                JJT = np.dot(jacobian, jacobian.T)
                manipulability = np.sqrt(np.linalg.det(JJT))
            else:
                manipulability = abs(np.linalg.det(jacobian))
        except:
            manipulability = 0.0
        
        return manipulability
    
    def get_transformation_matrices(self, dh_params, joint_angles):
        """각 링크의 변환 행렬들을 반환 (시각화용)
        
        Args:
            dh_params (list): DH 파라미터
            joint_angles (list): 관절 각도 (라디안)
            
        Returns:
            list: 각 링크의 누적 변환 행렬 리스트
        """
        transformation_matrices = [np.eye(4)]
        T_cumulative = np.eye(4)
        
        for i, (dh_param, joint_angle) in enumerate(zip(dh_params, joint_angles)):
            a, alpha_deg, d, theta_offset_deg = dh_param
            alpha = np.radians(alpha_deg)
            theta_offset = np.radians(theta_offset_deg)
            theta = joint_angle + theta_offset
            
            T_i = self.dh_transform_matrix(a, alpha, d, theta)
            T_cumulative = np.dot(T_cumulative, T_i)
            transformation_matrices.append(T_cumulative.copy())
        
        return transformation_matrices