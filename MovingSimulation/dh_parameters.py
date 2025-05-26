"""
dh_parameters.py - MovingSimulation/dh_parameters.py

DH 파라미터 관리 모듈
- 다양한 DOF 로봇의 기본 DH 파라미터 제공
- YAML 파일 저장/로드 기능
- 표준 로봇 구조의 DH 파라미터 예시
"""

import yaml
import numpy as np
import os

class DHParameterManager:
    def __init__(self):
        """DH 파라미터 관리자 초기화"""
        # 표준 로봇 구조의 DH 파라미터 데이터베이스
        self.standard_robots = {
            "1DOF_Revolute": {
                "description": "Simple 1-DOF revolute joint robot",
                "dh_params": [[30.0, 0.0, 0.0, 0.0]]  # [a, alpha, d, theta]
            },
            "2DOF_Planar": {
                "description": "2-DOF planar manipulator (like SCARA arm)",
                "dh_params": [
                    [40.0, 0.0, 0.0, 0.0],    # Link 1
                    [30.0, 0.0, 0.0, 0.0]     # Link 2
                ]
            },
            "3DOF_Anthropomorphic": {
                "description": "3-DOF anthropomorphic arm",
                "dh_params": [
                    [0.0, 90.0, 15.0, 0.0],   # Base rotation
                    [35.0, 0.0, 0.0, 0.0],    # Upper arm
                    [25.0, 0.0, 0.0, 0.0]     # Forearm
                ]
            },
            "4DOF_SCARA": {
                "description": "4-DOF SCARA robot",
                "dh_params": [
                    [35.0, 0.0, 20.0, 0.0],   # Link 1
                    [25.0, 180.0, 0.0, 0.0],  # Link 2
                    [0.0, 0.0, 15.0, 0.0],    # Vertical axis
                    [0.0, 0.0, 0.0, 0.0]      # End-effector rotation
                ]
            },
            "5DOF_Articulated": {
                "description": "5-DOF articulated robot arm",
                "dh_params": [
                    [0.0, 90.0, 18.0, 0.0],   # Base
                    [30.0, 0.0, 0.0, 0.0],    # Shoulder
                    [25.0, 0.0, 0.0, 0.0],    # Elbow
                    [0.0, 90.0, 20.0, 0.0],   # Wrist pitch
                    [0.0, 0.0, 8.0, 0.0]      # Wrist roll
                ]
            },
            "6DOF_Industrial": {
                "description": "6-DOF industrial robot (like PUMA-style)",
                "dh_params": [
                    [0.0, 90.0, 15.0, 0.0],   # Base rotation
                    [25.0, 0.0, 0.0, 0.0],    # Shoulder
                    [5.0, 90.0, 0.0, 0.0],    # Elbow
                    [0.0, -90.0, 22.0, 0.0],  # Wrist yaw
                    [0.0, 90.0, 0.0, 0.0],    # Wrist pitch
                    [0.0, 0.0, 6.0, 0.0]      # Wrist roll
                ]
            }
        }
        
        # 각 DOF별 기본 DH 파라미터 (간단한 구조)
        self.default_params = {
            1: [[25.0, 0.0, 0.0, 0.0]],
            2: [[30.0, 0.0, 0.0, 0.0], [25.0, 0.0, 0.0, 0.0]],
            3: [[0.0, 90.0, 12.0, 0.0], [25.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]],
            4: [[25.0, 0.0, 15.0, 0.0], [20.0, 180.0, 0.0, 0.0], [0.0, 0.0, 12.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            5: [[0.0, 90.0, 15.0, 0.0], [25.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0], [0.0, 90.0, 15.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
            6: [[0.0, 90.0, 12.0, 0.0], [20.0, 0.0, 0.0, 0.0], [3.0, 90.0, 0.0, 0.0], [0.0, -90.0, 18.0, 0.0], [0.0, 90.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]
        }
        
    def get_default_dh_params(self, dof):
        """
        지정된 DOF에 대한 기본 DH 파라미터 반환
        
        Args:
            dof (int): 자유도 (1-6)
            
        Returns:
            list: DH 파라미터 리스트
        """
        if dof in self.default_params:
            return self.default_params[dof].copy()
        else:
            # DOF가 지원 범위를 벗어나는 경우 기본값 생성
            return self.generate_default_params(dof)
    
    def generate_default_params(self, dof):
        """
        임의의 DOF에 대한 기본 DH 파라미터 생성
        
        Args:
            dof (int): 자유도
            
        Returns:
            list: 생성된 DH 파라미터 리스트
        """
        params = []
        base_length = 20.0  # 기본 링크 길이 (cm)
        
        for i in range(dof):
            if i == 0:  # 첫 번째 관절 (베이스)
                if dof > 2:
                    # 3DOF 이상은 베이스 회전 고려
                    a, alpha, d, theta = 0.0, 90.0, 10.0, 0.0
                else:
                    # 2DOF 이하는 평면 로봇
                    a, alpha, d, theta = base_length, 0.0, 0.0, 0.0
            elif i == dof - 1:  # 마지막 관절 (end-effector)
                a, alpha, d, theta = base_length * 0.6, 0.0, 0.0, 0.0
            elif i == dof - 2 and dof >= 4:  # 손목 관절
                a, alpha, d, theta = 0.0, 90.0, base_length * 0.8, 0.0
            else:  # 중간 관절들
                length_factor = 1.0 - (i / dof) * 0.3  # 점진적으로 짧아짐
                a, alpha, d, theta = base_length * length_factor, 0.0, 0.0, 0.0
            
            params.append([a, alpha, d, theta])
        
        return params
    
    def get_standard_robot_params(self, robot_name):
        """
        표준 로봇의 DH 파라미터 반환
        
        Args:
            robot_name (str): 로봇 이름
            
        Returns:
            dict: 로봇 정보 (description, dh_params)
        """
        return self.standard_robots.get(robot_name, None)
    
    def get_available_robots(self):
        """
        사용 가능한 표준 로봇 목록 반환
        
        Returns:
            list: 로봇 이름 리스트
        """
        return list(self.standard_robots.keys())
    
    def save_to_yaml(self, dh_params, file_path, dof=None, description="Custom robot"):
        """
        DH 파라미터를 YAML 파일로 저장
        
        Args:
            dh_params (list): DH 파라미터 리스트
            file_path (str): 저장할 파일 경로
            dof (int): 자유도
            description (str): 로봇 설명
        """
        try:
            # YAML 데이터 구조 생성
            data = {
                'robot_info': {
                    'name': os.path.splitext(os.path.basename(file_path))[0],
                    'description': description,
                    'dof': dof if dof else len(dh_params),
                    'created_date': self._get_current_date()
                },
                'dh_parameters': {
                    'format': 'a (cm), alpha (deg), d (cm), theta (deg)',
                    'links': []
                }
            }
            
            # 각 링크의 DH 파라미터 추가
            for i, params in enumerate(dh_params):
                a, alpha, d, theta = params
                link_data = {
                    'link_number': i + 1,
                    'a': float(a),
                    'alpha': float(alpha),
                    'd': float(d),
                    'theta': float(theta),
                    'description': f'Link {i + 1} parameters'
                }
                data['dh_parameters']['links'].append(link_data)
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, allow_unicode=True, 
                         sort_keys=False, indent=2)
            
            print(f"DH parameters saved to: {file_path}")
            
        except Exception as e:
            raise Exception(f"Error saving YAML file: {str(e)}")
    
    def load_from_yaml(self, file_path):
        """
        YAML 파일에서 DH 파라미터 로드
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            list: DH 파라미터 리스트 또는 None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            if 'dh_parameters' in data and 'links' in data['dh_parameters']:
                dh_params = []
                
                # 링크 번호 순으로 정렬
                links = sorted(data['dh_parameters']['links'], 
                             key=lambda x: x.get('link_number', 0))
                
                for link in links:
                    a = link.get('a', 0.0)
                    alpha = link.get('alpha', 0.0)
                    d = link.get('d', 0.0)
                    theta = link.get('theta', 0.0)
                    dh_params.append([a, alpha, d, theta])
                
                print(f"DH parameters loaded from: {file_path}")
                print(f"Robot: {data.get('robot_info', {}).get('name', 'Unknown')}")
                print(f"DOF: {len(dh_params)}")
                
                return dh_params
            else:
                raise Exception("Invalid YAML format: missing dh_parameters or links")
                
        except FileNotFoundError:
            raise Exception(f"File not found: {file_path}")
        except yaml.YAMLError as e:
            raise Exception(f"YAML parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading YAML file: {str(e)}")
    
    def create_default_yaml_files(self):
        """
        기본 YAML 파일들을 생성 (예시용)
        """
        yaml_dir = "./yaml"
        os.makedirs(yaml_dir, exist_ok=True)
        
        # 각 표준 로봇에 대한 YAML 파일 생성
        for robot_name, robot_data in self.standard_robots.items():
            file_path = os.path.join(yaml_dir, f"{robot_name.lower()}.yaml")
            
            # 파일이 이미 존재하면 건너뛰기
            if os.path.exists(file_path):
                continue
            
            try:
                self.save_to_yaml(
                    robot_data['dh_params'], 
                    file_path,
                    len(robot_data['dh_params']),
                    robot_data['description']
                )
            except Exception as e:
                print(f"Warning: Could not create {file_path}: {str(e)}")
    
    def validate_dh_params(self, dh_params):
        """
        DH 파라미터의 유효성 검사
        
        Args:
            dh_params (list): DH 파라미터 리스트
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(dh_params, list):
            return False, "DH parameters must be a list"
        
        if len(dh_params) == 0:
            return False, "DH parameters list is empty"
        
        for i, params in enumerate(dh_params):
            if not isinstance(params, (list, tuple)) or len(params) != 4:
                return False, f"Link {i+1}: Each parameter set must have exactly 4 values [a, alpha, d, theta]"
            
            try:
                # 숫자 변환 가능성 검사
                a, alpha, d, theta = [float(p) for p in params]
                
                # 합리적인 범위 검사
                if abs(a) > 200:  # 링크 길이가 2m보다 크면 경고
                    return False, f"Link {i+1}: Link length 'a' seems too large: {a} cm"
                
                if abs(alpha) > 360:  # 각도가 360도보다 크면 경고
                    return False, f"Link {i+1}: Link twist 'alpha' should be within ±360°: {alpha}°"
                
                if abs(d) > 200:  # 오프셋이 2m보다 크면 경고  
                    return False, f"Link {i+1}: Link offset 'd' seems too large: {d} cm"
                
                if abs(theta) > 360:  # 관절 오프셋이 360도보다 크면 경고
                    return False, f"Link {i+1}: Joint offset 'theta' should be within ±360°: {theta}°"
                    
            except (ValueError, TypeError):
                return False, f"Link {i+1}: All parameters must be numeric values"
        
        return True, "Valid DH parameters"
    
    def optimize_dh_params(self, dh_params, workspace_target=None):
        """
        작업공간 최적화를 위한 DH 파라미터 조정
        
        Args:
            dh_params (list): 원본 DH 파라미터
            workspace_target (dict): 목표 작업공간 정보
            
        Returns:
            list: 최적화된 DH 파라미터
        """
        # 간단한 최적화: 링크 길이 비율 조정
        optimized_params = []
        total_reach = sum([abs(param[0]) for param in dh_params])  # 총 도달거리
        
        if workspace_target and 'max_reach' in workspace_target:
            target_reach = workspace_target['max_reach']
            scale_factor = target_reach / total_reach if total_reach > 0 else 1.0
            
            for params in dh_params:
                a, alpha, d, theta = params
                # 링크 길이만 스케일링
                optimized_params.append([a * scale_factor, alpha, d, theta])
        else:
            optimized_params = dh_params.copy()
        
        return optimized_params
    
    def generate_random_params(self, dof, link_length_range=(10, 50), 
                             angle_range=(-90, 90)):
        """
        랜덤 DH 파라미터 생성
        
        Args:
            dof (int): 자유도
            link_length_range (tuple): 링크 길이 범위 (cm)
            angle_range (tuple): 각도 범위 (도)
            
        Returns:
            list: 랜덤 DH 파라미터
        """
        random_params = []
        
        for i in range(dof):
            # 링크 길이 (a)
            a = np.random.uniform(link_length_range[0], link_length_range[1])
            
            # 링크 트위스트 (alpha) - 주로 0, 90, -90도 중 선택
            alpha_choices = [0, 90, -90] + list(range(angle_range[0], angle_range[1], 30))
            alpha = np.random.choice(alpha_choices)
            
            # 링크 오프셋 (d)
            d = np.random.uniform(0, link_length_range[1] * 0.5)
            
            # 관절 오프셋 (theta) - 보통 0
            theta = 0  # 관절 변수이므로 초기값은 0
            
            random_params.append([round(a, 1), float(alpha), round(d, 1), float(theta)])
        
        return random_params
    
    def compare_robots(self, params1, params2, names=["Robot 1", "Robot 2"]):
        """
        두 로봇의 DH 파라미터 비교
        
        Args:
            params1 (list): 첫 번째 로봇의 DH 파라미터
            params2 (list): 두 번째 로봇의 DH 파라미터
            names (list): 로봇 이름들
            
        Returns:
            dict: 비교 결과
        """
        comparison = {
            'robots': names,
            'dof_comparison': (len(params1), len(params2)),
            'reach_comparison': {},
            'complexity_comparison': {}
        }
        
        # 최대 도달거리 비교
        reach1 = sum([abs(p[0]) for p in params1])  # 총 링크 길이
        reach2 = sum([abs(p[0]) for p in params2])
        comparison['reach_comparison'] = {
            names[0]: reach1,
            names[1]: reach2,
            'difference': abs(reach1 - reach2)
        }
        
        # 구조 복잡도 비교 (0이 아닌 alpha 각도의 개수)
        complex1 = sum([1 for p in params1 if abs(p[1]) > 1])
        complex2 = sum([1 for p in params2 if abs(p[1]) > 1])
        comparison['complexity_comparison'] = {
            names[0]: complex1,
            names[1]: complex2,
            'difference': abs(complex1 - complex2)
        }
        
        return comparison
    
    def export_to_csv(self, dh_params, file_path, robot_name="Custom Robot"):
        """
        DH 파라미터를 CSV 파일로 내보내기
        
        Args:
            dh_params (list): DH 파라미터
            file_path (str): CSV 파일 경로
            robot_name (str): 로봇 이름
        """
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # 헤더 작성
                writer.writerow(['Robot Name', robot_name])
                writer.writerow(['DOF', len(dh_params)])
                writer.writerow(['Created Date', self._get_current_date()])
                writer.writerow([])  # 빈 줄
                writer.writerow(['Link', 'a (cm)', 'alpha (deg)', 'd (cm)', 'theta (deg)'])
                
                # 데이터 작성
                for i, params in enumerate(dh_params):
                    writer.writerow([f'Link {i+1}'] + params)
            
            print(f"DH parameters exported to CSV: {file_path}")
            
        except Exception as e:
            raise Exception(f"Error exporting to CSV: {str(e)}")
    
    def import_from_csv(self, file_path):
        """
        CSV 파일에서 DH 파라미터 가져오기
        
        Args:
            file_path (str): CSV 파일 경로
            
        Returns:
            list: DH 파라미터 리스트
        """
        try:
            import csv
            dh_params = []
            
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                
                # DH 파라미터 데이터 찾기 (헤더 행 이후)
                data_start = -1
                for i, row in enumerate(rows):
                    if len(row) > 0 and 'Link' in row[0] and 'a (' in row[1]:
                        data_start = i + 1
                        break
                
                if data_start == -1:
                    raise Exception("Could not find DH parameter data in CSV")
                
                # 데이터 행 읽기
                for i in range(data_start, len(rows)):
                    row = rows[i]
                    if len(row) >= 5:  # Link, a, alpha, d, theta
                        try:
                            a = float(row[1])
                            alpha = float(row[2])
                            d = float(row[3])
                            theta = float(row[4])
                            dh_params.append([a, alpha, d, theta])
                        except ValueError:
                            continue  # 숫자가 아닌 행은 건너뛰기
            
            print(f"DH parameters imported from CSV: {file_path}")
            print(f"Loaded {len(dh_params)} links")
            
            return dh_params
            
        except Exception as e:
            raise Exception(f"Error importing from CSV: {str(e)}")
    
    def _get_current_date(self):
        """현재 날짜 문자열 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_dh_table(self, dh_params, robot_name="Robot"):
        """
        DH 파라미터를 테이블 형태로 출력
        
        Args:
            dh_params (list): DH 파라미터
            robot_name (str): 로봇 이름
        """
        print(f"\n=== {robot_name} DH Parameters ===")
        print(f"DOF: {len(dh_params)}")
        print("-" * 60)
        print(f"{'Link':<6} {'a (cm)':<10} {'α (deg)':<10} {'d (cm)':<10} {'θ (deg)':<10}")
        print("-" * 60)
        
        for i, params in enumerate(dh_params):
            a, alpha, d, theta = params
            print(f"{i+1:<6} {a:<10.1f} {alpha:<10.1f} {d:<10.1f} {theta:<10.1f}")
        
        print("-" * 60)
        
        # 추가 정보
        total_reach = sum([abs(p[0]) for p in dh_params])
        non_zero_alpha = sum([1 for p in dh_params if abs(p[1]) > 1])
        
        print(f"Total Reach: {total_reach:.1f} cm")
        print(f"Complex Joints (non-zero α): {non_zero_alpha}")
        print("")