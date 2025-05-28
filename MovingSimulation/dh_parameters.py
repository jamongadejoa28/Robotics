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
                "dh_params": [[30.0, 0.0, 0.0, 0.0]]
            },
            "2DOF_Planar": {
                "description": "2-DOF planar manipulator (like SCARA arm)",
                "dh_params": [
                    [40.0, 0.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0, 0.0]
                ]
            },
            "3DOF_Anthropomorphic": {
                "description": "3-DOF anthropomorphic arm",
                "dh_params": [
                    [0.0, 90.0, 15.0, 0.0],
                    [35.0, 0.0, 0.0, 0.0],
                    [25.0, 0.0, 0.0, 0.0]
                ]
            },
            "4DOF_SCARA": {
                "description": "4-DOF SCARA robot",
                "dh_params": [
                    [35.0, 0.0, 20.0, 0.0],
                    [25.0, 180.0, 0.0, 0.0],
                    [0.0, 0.0, 15.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0]
                ]
            },
            "5DOF_Articulated": {
                "description": "5-DOF articulated robot arm",
                "dh_params": [
                    [0.0, 90.0, 18.0, 0.0],
                    [30.0, 0.0, 0.0, 0.0],
                    [25.0, 0.0, 0.0, 0.0],
                    [0.0, 90.0, 20.0, 0.0],
                    [0.0, 0.0, 8.0, 0.0]
                ]
            },
            "6DOF_Industrial": {
                "description": "6-DOF industrial robot (like PUMA-style)",
                "dh_params": [
                    [0.0, 90.0, 15.0, 0.0],
                    [25.0, 0.0, 0.0, 0.0],
                    [5.0, 90.0, 0.0, 0.0],
                    [0.0, -90.0, 22.0, 0.0],
                    [0.0, 90.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 0.0]
                ]
            }
        }
        
        # 각 DOF별 기본 DH 파라미터
        self.default_params = {
            1: [[25.0, 0.0, 0.0, 0.0]],
            2: [[30.0, 0.0, 0.0, 0.0], [25.0, 0.0, 0.0, 0.0]],
            3: [[0.0, 90.0, 12.0, 0.0], [25.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]],
            4: [[25.0, 0.0, 15.0, 0.0], [20.0, 180.0, 0.0, 0.0], [0.0, 0.0, 12.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            5: [[0.0, 90.0, 15.0, 0.0], [25.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0], [0.0, 90.0, 15.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
            6: [[0.0, 90.0, 12.0, 0.0], [20.0, 0.0, 0.0, 0.0], [3.0, 90.0, 0.0, 0.0], [0.0, -90.0, 18.0, 0.0], [0.0, 90.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]]
        }
        
    def get_default_dh_params(self, dof):
        """지정된 DOF에 대한 기본 DH 파라미터 반환"""
        if dof in self.default_params:
            return self.default_params[dof].copy()
        else:
            return self.generate_default_params(dof)
    
    def generate_default_params(self, dof):
        """임의의 DOF에 대한 기본 DH 파라미터 생성"""
        params = []
        base_length = 20.0
        
        for i in range(dof):
            if i == 0:
                if dof > 2:
                    a, alpha, d, theta = 0.0, 90.0, 10.0, 0.0
                else:
                    a, alpha, d, theta = base_length, 0.0, 0.0, 0.0
            elif i == dof - 1:
                a, alpha, d, theta = base_length * 0.6, 0.0, 0.0, 0.0
            elif i == dof - 2 and dof >= 4:
                a, alpha, d, theta = 0.0, 90.0, base_length * 0.8, 0.0
            else:
                length_factor = 1.0 - (i / dof) * 0.3
                a, alpha, d, theta = base_length * length_factor, 0.0, 0.0, 0.0
            
            params.append([a, alpha, d, theta])
        
        return params
    
    def get_standard_robot_params(self, robot_name):
        """표준 로봇의 DH 파라미터 반환"""
        return self.standard_robots.get(robot_name, None)
    
    def get_available_robots(self):
        """사용 가능한 표준 로봇 목록 반환"""
        return list(self.standard_robots.keys())
    
    def save_to_yaml(self, dh_params, file_path, dof=None, description="Custom robot"):
        """DH 파라미터를 YAML 파일로 저장"""
        try:
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
            
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, allow_unicode=True, 
                         sort_keys=False, indent=2)
            
            print(f"DH parameters saved to: {file_path}")
            
        except Exception as e:
            raise Exception(f"Error saving YAML file: {str(e)}")
    
    def load_from_yaml(self, file_path):
        """YAML 파일에서 DH 파라미터 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            
            if 'dh_parameters' in data and 'links' in data['dh_parameters']:
                dh_params = []
                
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
        """기본 YAML 파일들을 생성"""
        yaml_dir = "./yaml"
        os.makedirs(yaml_dir, exist_ok=True)
        
        for robot_name, robot_data in self.standard_robots.items():
            file_path = os.path.join(yaml_dir, f"{robot_name.lower()}.yaml")
            
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
        """DH 파라미터의 유효성 검사"""
        if not isinstance(dh_params, list):
            return False, "DH parameters must be a list"
        
        if len(dh_params) == 0:
            return False, "DH parameters list is empty"
        
        for i, params in enumerate(dh_params):
            if not isinstance(params, (list, tuple)) or len(params) != 4:
                return False, f"Link {i+1}: Each parameter set must have exactly 4 values [a, alpha, d, theta]"
            
            try:
                a, alpha, d, theta = [float(p) for p in params]
                
                if abs(a) > 200:
                    return False, f"Link {i+1}: Link length 'a' seems too large: {a} cm"
                
                if abs(alpha) > 360:
                    return False, f"Link {i+1}: Link twist 'alpha' should be within ±360°: {alpha}°"
                
                if abs(d) > 200:
                    return False, f"Link {i+1}: Link offset 'd' seems too large: {d} cm"
                
                if abs(theta) > 360:
                    return False, f"Link {i+1}: Joint offset 'theta' should be within ±360°: {theta}°"
                    
            except (ValueError, TypeError):
                return False, f"Link {i+1}: All parameters must be numeric values"
        
        return True, "Valid DH parameters"
    
    def optimize_dh_params(self, dh_params, workspace_target=None):
        """작업공간 최적화를 위한 DH 파라미터 조정"""
        optimized_params = []
        total_reach = sum([abs(param[0]) for param in dh_params])
        
        if workspace_target and 'max_reach' in workspace_target:
            target_reach = workspace_target['max_reach']
            scale_factor = target_reach / total_reach if total_reach > 0 else 1.0
            
            for params in dh_params:
                a, alpha, d, theta = params
                optimized_params.append([a * scale_factor, alpha, d, theta])
        else:
            optimized_params = dh_params.copy()
        
        return optimized_params
    
    def generate_random_params(self, dof, link_length_range=(10, 50), 
                             angle_range=(-90, 90)):
        """랜덤 DH 파라미터 생성"""
        random_params = []
        
        for i in range(dof):
            a = np.random.uniform(link_length_range[0], link_length_range[1])
            
            alpha_choices = [0, 90, -90] + list(range(angle_range[0], angle_range[1], 30))
            alpha = np.random.choice(alpha_choices)
            
            d = np.random.uniform(0, link_length_range[1] * 0.5)
            
            theta = 0
            
            random_params.append([round(a, 1), float(alpha), round(d, 1), float(theta)])
        
        return random_params
    
    def compare_robots(self, params1, params2, names=["Robot 1", "Robot 2"]):
        """두 로봇의 DH 파라미터 비교"""
        comparison = {
            'robots': names,
            'dof_comparison': (len(params1), len(params2)),
            'reach_comparison': {},
            'complexity_comparison': {}
        }
        
        reach1 = sum([abs(p[0]) for p in params1])
        reach2 = sum([abs(p[0]) for p in params2])
        comparison['reach_comparison'] = {
            names[0]: reach1,
            names[1]: reach2,
            'difference': abs(reach1 - reach2)
        }
        
        complex1 = sum([1 for p in params1 if abs(p[1]) > 1])
        complex2 = sum([1 for p in params2 if abs(p[1]) > 1])
        comparison['complexity_comparison'] = {
            names[0]: complex1,
            names[1]: complex2,
            'difference': abs(complex1 - complex2)
        }
        
        return comparison
    
    def export_to_csv(self, dh_params, file_path, robot_name="Custom Robot"):
        """DH 파라미터를 CSV 파일로 내보내기"""
        try:
            import csv
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow(['Robot Name', robot_name])
                writer.writerow(['DOF', len(dh_params)])
                writer.writerow(['Created Date', self._get_current_date()])
                writer.writerow([])
                writer.writerow(['Link', 'a (cm)', 'alpha (deg)', 'd (cm)', 'theta (deg)'])
                
                for i, params in enumerate(dh_params):
                    writer.writerow([f'Link {i+1}'] + params)
            
            print(f"DH parameters exported to CSV: {file_path}")
            
        except Exception as e:
            raise Exception(f"Error exporting to CSV: {str(e)}")
    
    def import_from_csv(self, file_path):
        """CSV 파일에서 DH 파라미터 가져오기"""
        try:
            import csv
            dh_params = []
            
            with open(file_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                
                data_start = -1
                for i, row in enumerate(rows):
                    if len(row) > 0 and 'Link' in row[0] and 'a (' in row[1]:
                        data_start = i + 1
                        break
                
                if data_start == -1:
                    raise Exception("Could not find DH parameter data in CSV")
                
                for i in range(data_start, len(rows)):
                    row = rows[i]
                    if len(row) >= 5:
                        try:
                            a = float(row[1])
                            alpha = float(row[2])
                            d = float(row[3])
                            theta = float(row[4])
                            dh_params.append([a, alpha, d, theta])
                        except ValueError:
                            continue
            
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
        """DH 파라미터를 테이블 형태로 출력"""
        print(f"\n=== {robot_name} DH Parameters ===")
        print(f"DOF: {len(dh_params)}")
        print("-" * 60)
        print(f"{'Link':<6} {'a (cm)':<10} {'α (deg)':<10} {'d (cm)':<10} {'θ (deg)':<10}")
        print("-" * 60)
        
        for i, params in enumerate(dh_params):
            a, alpha, d, theta = params
            print(f"{i+1:<6} {a:<10.1f} {alpha:<10.1f} {d:<10.1f} {theta:<10.1f}")
        
        print("-" * 60)
        
        total_reach = sum([abs(p[0]) for p in dh_params])
        non_zero_alpha = sum([1 for p in dh_params if abs(p[1]) > 1])
        
        print(f"Total Reach: {total_reach:.1f} cm")
        print(f"Complex Joints (non-zero α): {non_zero_alpha}")
        print("")