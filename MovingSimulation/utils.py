"""
utils.py - MovingSimulation/utils.py

유틸리티 모듈
- 파일 입출력 도구
- 데이터 변환 함수
- 수학적 유틸리티
- 검증 및 오류 처리
- 성능 측정 도구
"""

import numpy as np
import csv
import json
import os
import time
from datetime import datetime
import warnings
from functools import wraps

class Utils:
    def __init__(self):
        """유틸리티 클래스 초기화"""
        # 각도 변환 상수
        self.DEG2RAD = np.pi / 180.0
        self.RAD2DEG = 180.0 / np.pi
        
        # 기본 허용 오차
        self.DEFAULT_TOLERANCE = 1e-6
        
        # 로그 설정
        self.enable_logging = True
        self.log_file = None
        
    def deg_to_rad(self, degrees):
        """
        도에서 라디안으로 변환
        
        Args:
            degrees (float or list): 각도(도)
            
        Returns:
            float or list: 각도(라디안)
        """
        if isinstance(degrees, (list, tuple, np.ndarray)):
            return [d * self.DEG2RAD for d in degrees]
        else:
            return degrees * self.DEG2RAD
    
    def rad_to_deg(self, radians):
        """
        라디안에서 도로 변환
        
        Args:
            radians (float or list): 각도(라디안)
            
        Returns:
            float or list: 각도(도)
        """
        if isinstance(radians, (list, tuple, np.ndarray)):
            return [r * self.RAD2DEG for r in radians]
        else:
            return radians * self.RAD2DEG
    
    def normalize_angle(self, angle, angle_range=(-np.pi, np.pi)):
        """
        각도를 지정된 범위로 정규화
        
        Args:
            angle (float): 입력 각도 (라디안)
            angle_range (tuple): 목표 범위 (min, max)
            
        Returns:
            float: 정규화된 각도
        """
        min_angle, max_angle = angle_range
        range_size = max_angle - min_angle
        
        # 각도를 [-π, π] 범위로 먼저 변환
        normalized = angle
        while normalized > max_angle:
            normalized -= range_size
        while normalized < min_angle:
            normalized += range_size
            
        return normalized
    
    def normalize_angle_degrees(self, angle, angle_range=(-180, 180)):
        """
        각도를 지정된 범위로 정규화 (도 단위)
        
        Args:
            angle (float): 입력 각도 (도)
            angle_range (tuple): 목표 범위 (min, max)
            
        Returns:
            float: 정규화된 각도
        """
        min_angle, max_angle = angle_range
        range_size = max_angle - min_angle
        
        normalized = angle
        while normalized > max_angle:
            normalized -= range_size
        while normalized < min_angle:
            normalized += range_size
            
        return normalized
    
    def rotation_matrix_x(self, angle):
        """
        X축 회전 행렬 생성
        
        Args:
            angle (float): 회전 각도 (라디안)
            
        Returns:
            np.array: 3x3 회전 행렬
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def rotation_matrix_y(self, angle):
        """
        Y축 회전 행렬 생성
        
        Args:
            angle (float): 회전 각도 (라디안)
            
        Returns:
            np.array: 3x3 회전 행렬
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def rotation_matrix_z(self, angle):
        """
        Z축 회전 행렬 생성
        
        Args:
            angle (float): 회전 각도 (라디안)
            
        Returns:
            np.array: 3x3 회전 행렬
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw, order='xyz'):
        """
        오일러 각을 회전 행렬로 변환
        
        Args:
            roll (float): 롤 각도 (라디안)
            pitch (float): 피치 각도 (라디안)
            yaw (float): 요 각도 (라디안)
            order (str): 회전 순서 ('xyz', 'zyx', etc.)
            
        Returns:
            np.array: 3x3 회전 행렬
        """
        Rx = self.rotation_matrix_x(roll)
        Ry = self.rotation_matrix_y(pitch)
        Rz = self.rotation_matrix_z(yaw)
        
        if order.lower() == 'xyz':
            return Rz @ Ry @ Rx
        elif order.lower() == 'zyx':
            return Rx @ Ry @ Rz
        elif order.lower() == 'zxy':
            return Ry @ Rx @ Rz
        else:
            raise ValueError(f"Unsupported rotation order: {order}")
    
    def rotation_matrix_to_euler(self, R, order='xyz'):
        """
        회전 행렬을 오일러 각으로 변환
        
        Args:
            R (np.array): 3x3 회전 행렬
            order (str): 회전 순서
            
        Returns:
            tuple: (roll, pitch, yaw) 라디안
        """
        if order.lower() == 'xyz':
            # ZYX 오일러 각 추출
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6
            
            if not singular:
                roll = np.arctan2(R[2, 1], R[2, 2])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = np.arctan2(R[1, 0], R[0, 0])
            else:
                roll = np.arctan2(-R[1, 2], R[1, 1])
                pitch = np.arctan2(-R[2, 0], sy)
                yaw = 0
                
            return roll, pitch, yaw
        else:
            raise ValueError(f"Unsupported rotation order: {order}")
    
    def homogeneous_transform(self, rotation_matrix, translation_vector):
        """
        회전 행렬과 평행이동 벡터로 동차 변환 행렬 생성
        
        Args:
            rotation_matrix (np.array): 3x3 회전 행렬
            translation_vector (np.array): 3x1 평행이동 벡터
            
        Returns:
            np.array: 4x4 동차 변환 행렬
        """
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation_vector
        return T
    
    def inverse_homogeneous_transform(self, T):
        """
        동차 변환 행렬의 역변환
        
        Args:
            T (np.array): 4x4 동차 변환 행렬
            
        Returns:
            np.array: 4x4 역변환 행렬
        """
        R = T[:3, :3]
        t = T[:3, 3]
        
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        
        return T_inv
    
    def is_valid_rotation_matrix(self, R, tolerance=None):
        """
        회전 행렬의 유효성 검사
        
        Args:
            R (np.array): 3x3 행렬
            tolerance (float): 허용 오차
            
        Returns:
            bool: 유효한 회전 행렬 여부
        """
        if tolerance is None:
            tolerance = self.DEFAULT_TOLERANCE
        
        # 직교성 검사: R * R^T = I
        should_be_identity = R @ R.T
        identity = np.eye(3)
        
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        
        # 행렬식 검사: det(R) = 1
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=tolerance):
            return False
        
        return True
    
    def interpolate_angles(self, angle1, angle2, t):
        """
        두 각도 사이의 선형 보간 (최단 경로)
        
        Args:
            angle1 (float): 시작 각도 (라디안)
            angle2 (float): 끝 각도 (라디안)  
            t (float): 보간 계수 (0~1)
            
        Returns:
            float: 보간된 각도
        """
        # 각도 차이 계산 (최단 경로)
        diff = self.normalize_angle(angle2 - angle1)
        
        # 선형 보간
        result = angle1 + t * diff
        
        return self.normalize_angle(result)
    
    def save_results_to_csv(self, data, file_path):
        """
        시뮬레이션 결과를 CSV 파일로 저장
        
        Args:
            data (dict): 저장할 데이터
            file_path (str): 파일 경로
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 메타데이터 작성
                writer.writerow(['Robot Simulation Results'])
                writer.writerow(['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow(['DOF', data.get('DOF', 'Unknown')])
                writer.writerow(['Robot Type', data.get('Robot_Type', 'Unknown')])
                writer.writerow([])  # 빈 줄
                
                # DH 파라미터 섹션
                writer.writerow(['DH Parameters'])
                writer.writerow(['Link', 'a (cm)', 'alpha (deg)', 'd (cm)', 'theta (deg)'])
                
                if 'DH_Parameters' in data:
                    for i, params in enumerate(data['DH_Parameters']):
                        writer.writerow([f'Link {i+1}'] + params)
                
                writer.writerow([])  # 빈 줄
                
                # 관절 각도 섹션
                writer.writerow(['Joint Angles'])
                writer.writerow(['Joint', 'Angle (deg)'])
                
                if 'Joint_Angles_deg' in data:
                    for i, angle in enumerate(data['Joint_Angles_deg']):
                        writer.writerow([f'Joint {i+1}', f'{angle:.3f}'])
                
                writer.writerow([])  # 빈 줄
                
                # End-effector 위치 섹션
                writer.writerow(['End-Effector Position'])
                writer.writerow(['Axis', 'Position (cm)'])
                
                if 'End_Effector_Position_cm' in data:
                    axes = ['X', 'Y', 'Z']
                    for i, pos in enumerate(data['End_Effector_Position_cm']):
                        if i < len(axes):
                            writer.writerow([axes[i], f'{pos:.3f}'])
                
            print(f"Results saved to: {file_path}")
            
        except Exception as e:
            raise Exception(f"Error saving CSV file: {str(e)}")
    
    def load_results_from_csv(self, file_path):
        """
        CSV 파일에서 시뮬레이션 결과 로드
        
        Args:
            file_path (str): 파일 경로
            
        Returns:
            dict: 로드된 데이터
        """
        try:
            data = {}
            
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                
                # 메타데이터 파싱
                for i, row in enumerate(rows):
                    if len(row) >= 2:
                        if row[0] == 'DOF':
                            data['DOF'] = int(row[1]) if row[1].isdigit() else row[1]
                        elif row[0] == 'Robot Type':
                            data['Robot_Type'] = row[1]
                
                # DH 파라미터 파싱
                dh_start = -1
                for i, row in enumerate(rows):
                    if len(row) > 0 and 'Link' in row[0] and 'a (' in str(row):
                        dh_start = i + 1
                        break
                
                if dh_start > 0:
                    dh_params = []
                    for i in range(dh_start, len(rows)):
                        row = rows[i]
                        if len(row) >= 5 and 'Link' in row[0]:
                            try:
                                params = [float(row[j]) for j in range(1, 5)]
                                dh_params.append(params)
                            except ValueError:
                                break
                        elif len(row) == 0:
                            break
                    data['DH_Parameters'] = dh_params
                
            return data
            
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def validate_joint_angles(self, joint_angles, joint_limits=None):
        """
        관절 각도 유효성 검사
        
        Args:
            joint_angles (list): 관절 각도 리스트 (라디안)
            joint_limits (dict): 관절 제한 {joint_idx: (min_deg, max_deg)}
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(joint_angles, (list, tuple, np.ndarray)):
            return False, "Joint angles must be a list, tuple, or numpy array"
        
        if len(joint_angles) == 0:
            return False, "Joint angles list is empty"
        
        # 숫자값 검사
        try:
            angles_array = np.array(joint_angles, dtype=float)
        except (ValueError, TypeError):
            return False, "All joint angles must be numeric values"
        
        # NaN 또는 무한대 검사
        if np.any(np.isnan(angles_array)) or np.any(np.isinf(angles_array)):
            return False, "Joint angles contain NaN or infinite values"
        
        # 관절 제한 검사
        if joint_limits:
            for i, angle_rad in enumerate(angles_array):
                if i in joint_limits:
                    min_deg, max_deg = joint_limits[i]
                    angle_deg = self.rad_to_deg(angle_rad)
                    
                    if angle_deg < min_deg or angle_deg > max_deg:
                        return False, f"Joint {i+1} angle {angle_deg:.1f}° exceeds limits [{min_deg}°, {max_deg}°]"
        
        return True, "Valid joint angles"
    
    def validate_dh_parameters(self, dh_params):
        """
        DH 파라미터 유효성 검사 (dh_parameters.py의 기능을 보완)
        
        Args:
            dh_params (list): DH 파라미터 리스트
            
        Returns:
            tuple: (is_valid, error_message, warnings)
        """
        if not isinstance(dh_params, list):
            return False, "DH parameters must be a list", []
        
        if len(dh_params) == 0:
            return False, "DH parameters list is empty", []
        
        warnings_list = []
        
        for i, params in enumerate(dh_params):
            if not isinstance(params, (list, tuple)) or len(params) != 4:
                return False, f"Link {i+1}: Each parameter set must have 4 values [a, alpha, d, theta]", warnings_list
            
            try:
                a, alpha, d, theta = [float(p) for p in params]
                
                # 경고 사항 검사
                if abs(a) > 500:  # 5m보다 긴 링크
                    warnings_list.append(f"Link {i+1}: Very long link length: {a} cm")
                
                if abs(d) > 500:  # 5m보다 긴 오프셋
                    warnings_list.append(f"Link {i+1}: Very large offset: {d} cm")
                
                if abs(alpha) > 180 and abs(alpha) != 180:  # 180도가 아닌 큰 각도
                    warnings_list.append(f"Link {i+1}: Unusual twist angle: {alpha}°")
                
            except (ValueError, TypeError):
                return False, f"Link {i+1}: All parameters must be numeric", warnings_list
        
        return True, "Valid DH parameters", warnings_list
    
    def measure_execution_time(self, func):
        """
        함수 실행 시간 측정 데코레이터
        
        Args:
            func: 측정할 함수
            
        Returns:
            wrapper function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            if self.enable_logging:
                print(f"{func.__name__} execution time: {execution_time:.4f} seconds")
            
            return result
        return wrapper
    
    def log_message(self, message, level='INFO'):
        """
        로그 메시지 기록
        
        Args:
            message (str): 로그 메시지
            level (str): 로그 레벨 ('INFO', 'WARNING', 'ERROR')
        """
        if not self.enable_logging:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        
        print(log_entry)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"Failed to write to log file: {e}")
    
    def set_log_file(self, file_path):
        """
        로그 파일 설정
        
        Args:
            file_path (str): 로그 파일 경로
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.log_file = file_path
            self.log_message(f"Log file set to: {file_path}")
        except Exception as e:
            print(f"Failed to set log file: {e}")
    
    def create_backup(self, source_file, backup_dir=None):
        """
        파일 백업 생성
        
        Args:
            source_file (str): 백업할 파일
            backup_dir (str): 백업 디렉토리 (기본값: source_file_backups)
            
        Returns:
            str: 백업 파일 경로
        """
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        if backup_dir is None:
            backup_dir = os.path.dirname(source_file) + "_backups"
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # 백업 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(source_file)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # 파일 복사
        import shutil
        shutil.copy2(source_file, backup_path)
        
        self.log_message(f"Backup created: {backup_path}")
        return backup_path
    
    def calculate_statistics(self, data_array):
        """
        데이터 배열의 통계 정보 계산
        
        Args:
            data_array (np.array): 데이터 배열
            
        Returns:
            dict: 통계 정보
        """
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)
        
        if data_array.size == 0:
            return {'error': 'Empty data array'}
        
        stats = {
            'count': data_array.size,
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'range': np.max(data_array) - np.min(data_array),
            'percentile_25': np.percentile(data_array, 25),
            'percentile_75': np.percentile(data_array, 75)
        }
        
        # 추가 통계 (0이 아닌 값들에 대해)
        non_zero_data = data_array[data_array != 0]
        if non_zero_data.size > 0:
            stats['non_zero_count'] = non_zero_data.size
            stats['non_zero_mean'] = np.mean(non_zero_data)
            stats['rms'] = np.sqrt(np.mean(data_array**2))
        
        return stats
    
    def format_number(self, number, decimals=3, scientific_threshold=1000):
        """
        숫자를 보기 좋게 포맷팅
        
        Args:
            number (float): 포맷팅할 숫자
            decimals (int): 소수점 자릿수
            scientific_threshold (float): 과학적 표기법 사용 임계값
            
        Returns:
            str: 포맷팅된 문자열
        """
        if abs(number) >= scientific_threshold or (abs(number) < 0.001 and number != 0):
            return f"{number:.{decimals}e}"
        else:
            return f"{number:.{decimals}f}"
    
    def safe_divide(self, numerator, denominator, default_value=0.0):
        """
        안전한 나눗셈 (0으로 나누기 방지)
        
        Args:
            numerator (float): 분자
            denominator (float): 분모
            default_value (float): 분모가 0일 때 반환값
            
        Returns:
            float: 나눗셈 결과 또는 기본값
        """
        if abs(denominator) < self.DEFAULT_TOLERANCE:
            return default_value
        else:
            return numerator / denominator
    
    def clamp(self, value, min_value, max_value):
        """
        값을 지정된 범위로 제한
        
        Args:
            value (float): 입력값
            min_value (float): 최소값
            max_value (float): 최대값
            
        Returns:
            float: 제한된 값
        """
        return max(min_value, min(max_value, value))
    
    def linear_map(self, value, from_range, to_range):
        """
        값을 한 범위에서 다른 범위로 선형 매핑
        
        Args:
            value (float): 입력값
            from_range (tuple): 입력 범위 (min, max)
            to_range (tuple): 출력 범위 (min, max)
            
        Returns:
            float: 매핑된 값
        """
        from_min, from_max = from_range
        to_min, to_max = to_range
        
        # 입력 범위에서의 비율 계산
        ratio = (value - from_min) / (from_max - from_min)
        
        # 출력 범위로 매핑
        mapped_value = to_min + ratio * (to_max - to_min)
        
        return mapped_value
    
    def moving_average(self, data, window_size):
        """
        이동 평균 계산
        
        Args:
            data (list or np.array): 입력 데이터
            window_size (int): 윈도우 크기
            
        Returns:
            np.array: 이동 평균 배열
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        data_array = np.array(data)
        if len(data_array) < window_size:
            return data_array
        
        # 컨볼루션을 이용한 이동 평균
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data_array, kernel, mode='valid')
        
        # 시작 부분 패딩 (원본 크기 유지)
        padding = np.full(window_size - 1, smoothed[0])
        result = np.concatenate([padding, smoothed])
        
        return result
    
    def find_peaks(self, data, threshold=None, min_distance=1):
        """
        데이터에서 피크 찾기
        
        Args:
            data (list or np.array): 입력 데이터
            threshold (float): 피크 임계값
            min_distance (int): 피크 간 최소 거리
            
        Returns:
            list: 피크 인덱스 리스트
        """
        data_array = np.array(data)
        
        if threshold is None:
            threshold = np.mean(data_array) + np.std(data_array)
        
        peaks = []
        for i in range(1, len(data_array) - 1):
            if (data_array[i] > data_array[i-1] and 
                data_array[i] > data_array[i+1] and 
                data_array[i] > threshold):
                
                # 최소 거리 조건 확인
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    def get_system_info(self):
        """
        시스템 정보 수집
        
        Returns:
            dict: 시스템 정보
        """
        import platform
        import psutil
        
        info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return info
    
    def create_progress_bar(self, total, width=50):
        """
        간단한 진행률 표시줄 생성기
        
        Args:
            total (int): 전체 작업 수
            width (int): 진행률 바 너비
            
        Returns:
            function: 업데이트 함수
        """
        def update_progress(current, message=""):
            percent = (current / total) * 100
            filled = int(width * current // total)
            bar = '█' * filled + '-' * (width - filled)
            
            print(f'\r|{bar}| {percent:.1f}% {message}', end='', flush=True)
            
            if current >= total:
                print()  # 완료 시 새 줄
        
        return update_progress