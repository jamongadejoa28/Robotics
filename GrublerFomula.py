# -*- coding: utf-8 -*-
import math

def calculate_dof_grubler(num_links, joint_details, lambda_val):
    """
    Calculates the Degrees of Freedom (DOF) of a mechanism using Grubler's Formula.
    M = lambda * (N - 1 - J_total) + sum(f_i)

    Args:
        num_links (int): Total number of links, including the ground/fixed link (N).
        joint_details (list of tuples): A list where each tuple is (count, freedom).
                                        'count' is the number of joints of a certain type,
                                        and 'freedom' is the degrees offreedom of that joint type.
        lambda_val (int): Mobility of the space. 3 for planar mechanisms, 6 for spatial mechanisms.

    Returns:
        int: The calculated degrees of freedom (M), or None if input is invalid.
    """
    if num_links <= 0:
        print("오류: 링크 수는 0보다 커야 합니다.")
        return None

    total_num_joints = 0
    sum_of_joint_freedoms = 0

    for count, freedom in joint_details:
        if count < 0 or freedom < 0:
            print(f"오류: 조인트 개수({count})와 자유도({freedom})는 음수가 될 수 없습니다.")
            return None
        if freedom > lambda_val: # A joint's freedom cannot exceed the space's mobility
            print(f"오류: 조인트 자유도({freedom})는 공간의 차원({lambda_val})보다 클 수 없습니다.")
            return None
        total_num_joints += count
        sum_of_joint_freedoms += count * freedom

    dof = lambda_val * (num_links - 1 - total_num_joints) + sum_of_joint_freedoms
    return dof

def get_int_input(prompt, min_val=None, max_val=None, zero_allowed=False):
    while True:
        try:
            value = int(input(prompt))
            if not zero_allowed and min_val is None and value <= 0:
                 print("입력값은 0보다 커야 합니다.")
                 continue
            if min_val is not None and value < min_val:
                print(f"입력값은 {min_val} 이상이어야 합니다.")
                continue
            if max_val is not None and value > max_val:
                print(f"입력값은 {max_val} 이하여야 합니다.")
                continue
            return value
        except ValueError:
            print("유효한 정수를 입력해주세요.")

def analyze_possible_configurations(N_total, joint_details_list, lambda_val, M_overall_calculated):
    """
    Analyzes and suggests possible configurations based on a simplified decomposition model.
    Model: k free moving links + one mechanism formed by the rest of the links and all joints.
    """
    print("\n--- 가능한 구성 추측 (단순 분해 모델 기반) ---")
    print("참고: 아래 추측은 시스템이 'k개의 자유 이동 링크'와 '나머지 링크 및 모든 조인트로 구성된 단일 메커니즘'으로 분해될 수 있다는 가정에 기반합니다.")

    num_moving_links_total = N_total - 1
    if num_moving_links_total < 0:
        num_moving_links_total = 0 # Handle N_total = 0 or 1

    J_total_count = sum(count for count, freedom in joint_details_list)
    sum_f_i_total = sum(count * freedom for count, freedom in joint_details_list)

    found_configs_count = 0

    # k_free_moving은 자유롭게 움직이는 '움직이는 링크'의 수
    for k_free_moving in range(num_moving_links_total + 1):
        dof_from_free_links = k_free_moving * lambda_val
        
        # 메커니즘 부분을 구성하는 링크 수 (고정 링크 포함)
        num_links_in_mech_part = N_total - k_free_moving
        
        # 이 메커니즘 부분이 가져야 할 기대 자유도
        expected_dof_of_mech_part = M_overall_calculated - dof_from_free_links

        # 유효성 검사:
        # 1. 메커니즘 부분에 링크가 적어도 하나(고정 링크)는 있어야 함
        if num_links_in_mech_part < 1:
            continue
        # 2. 메커니즘 부분에 고정 링크만 남았는데 (num_links_in_mech_part == 1),
        #    조인트가 있다고 가정하는 것은 이 단순 모델에 부적합
        if num_links_in_mech_part == 1 and J_total_count > 0:
            continue
        # 3. 메커니즘 부분에 움직이는 링크가 없는데 (num_links_in_mech_part == 1),
        #    기대 자유도가 0이 아니면 모순
        if num_links_in_mech_part == 1 and J_total_count == 0 and abs(expected_dof_of_mech_part) > 1e-9 : # 거의 0이 아니면
            continue


        # 이 "메커니즘 부분"의 자유도를 그루블러로 직접 계산
        # (N_m, J_m, sum_f_m) = (num_links_in_mech_part, J_total_count, sum_f_i_total)
        # 모든 조인트가 이 메커니즘 부분에 사용된다고 가정
        
        calculated_dof_of_mech_part = 0 # 기본값
        if num_links_in_mech_part == 1 and J_total_count == 0 : # 고정 링크만, 조인트 없음
             calculated_dof_of_mech_part = 0
        elif num_links_in_mech_part > 1 : # 조인트가 있든 없든, 링크 2개 이상
            # 조인트가 없는 경우 (J_total_count == 0), 이 링크들은 사실상 추가적인 자유 링크임
            # 이 경우는 k_free_moving이 num_moving_links_total일 때와 동일하게 처리됨
            if J_total_count == 0:
                # (N_total - k_free_moving - 1) 개의 움직이는 링크가 추가로 자유로움.
                # 이는 k_free_moving이 모든 움직이는 링크를 포함하는 시나리오에서 다뤄짐.
                # 예를 들어 N=5, k=0, J=0 이면, M=3*(5-1)=12. 여기서 N_mech=5, J_mech=0. M_calc_mech = 3*(5-1-0)+0 = 12.
                pass # 아래에서 계산됨
            
            # 조인트가 0개라도 아래 공식은 유효 (예: N_m개의 링크, 0개 조인트 => 3*(N_m-1) DOF)
            calculated_dof_of_mech_part = lambda_val * (num_links_in_mech_part - 1 - J_total_count) + sum_f_i_total
        else: # num_links_in_mech_part < 1 이거나, num_links_in_mech_part == 1 인데 J_total_count > 0 인 경우 (위에서 continue됨)
            continue


        # 직접 계산한 메커니즘 DOF와 기대 DOF가 일치하는지 확인
        if abs(calculated_dof_of_mech_part - expected_dof_of_mech_part) < 1e-9:
            found_configs_count += 1
            description = f"  추측 {found_configs_count}: "
            
            if k_free_moving > 0:
                description += f"{k_free_moving}개의 자유 이동 링크 (각 {lambda_val}DOF, 총 {dof_from_free_links}DOF)"
            
            if num_links_in_mech_part == 1 and J_total_count == 0: # 고정 링크만 남은 경우
                if k_free_moving > 0: description += " + "
                description += "1개의 고정 링크 (0DOF 메커니즘)"
            elif J_total_count == 0 and num_links_in_mech_part > 1: # 조인트 없이 여러 링크 (사실상 자유 링크 그룹)
                 # 이 경우는 k_free_moving = N_total - 1 일때와 결과적으로 동일함.
                 # 예를 들어, k_free_moving = 0 이고 N_total=5, J_total=0 이면, M_overall = 12.
                 # N_mech=5, J_mech=0. calc_M_mech=12. exp_M_mech=12.
                 # "0개의 자유링크 + 5개 링크와 0개 조인트로 구성된 메커니즘 (12 DOF)"
                 if k_free_moving > 0: description += " + "
                 description += (f"{num_links_in_mech_part}개 링크와 0개 조인트로 구성된 부분 "
                                 f"(모든 내부 링크가 자유로워 총 {calculated_dof_of_mech_part}DOF)")
            elif J_total_count > 0 and num_links_in_mech_part >= 2: # 일반적인 메커니즘 부분
                if k_free_moving > 0: description += " + "
                description += (f"{num_links_in_mech_part}개 링크와 {J_total_count}개 조인트로 구성된 메커니즘 "
                                f"(DOF: {calculated_dof_of_mech_part})")
                # 예시 구조 추가
                if calculated_dof_of_mech_part > 0:
                    if num_links_in_mech_part == J_total_count + 1: # 대략적인 열린 체인 조건
                        description += " (예: 열린 체인 구조)"
                    elif num_links_in_mech_part == J_total_count and num_links_in_mech_part >=3 : # 대략적인 단일 폐쇄 루프 조건
                        description += " (예: 단일 폐쇄 루프 구조)"
            elif J_total_count > 0 and num_links_in_mech_part < 2 : # 조인트는 있는데 링크가 부족 (이론상 이전에 걸러져야 함)
                continue # 스킵

            print(description + ".")

    if found_configs_count == 0:
        print("  주어진 조건에 대한 위의 단순 분해 모델로는 명확한 구성을 제시하기 어렵습니다.")


def get_user_input_and_calculate_dof():
    print("--- 그루블러 공식을 이용한 로봇 자유도 계산 ---")
    lambda_val = 0
    while True:
        choice = input("메커니즘은 평면형입니까, 공간형입니까? ('평면형' 또는 '공간형' 입력): ").strip().lower()
        if choice == '평면형':
            lambda_val = 3
            break
        elif choice == '공간형':
            lambda_val = 6
            break
        else:
            print("잘못된 입력입니다. '평면형' 또는 '공간형'으로 입력해주세요.")

    num_links = get_int_input("총 링크 수 (고정 링크 포함)를 입력하세요: ", min_val=1)

    joint_details = []
    num_joint_types = get_int_input("서로 다른 종류의 조인트가 몇 개 있습니까? (예: 1자유도 조인트와 2자유도 조인트가 있다면 2종류): ", min_val=0, zero_allowed=True)

    for i in range(num_joint_types):
        print(f"\n--- 조인트 타입 {i+1} ---")
        count = get_int_input(f"이 타입의 조인트는 몇 개입니까?: ", min_val=0, zero_allowed=True)
        if count == 0:
            continue
        freedom = get_int_input(f"이 조인트들의 각 자유도는 얼마입니까? (예: 회전/직선은 1, 원통형은 2): ", min_val=1, max_val=lambda_val)
        joint_details.append((count, freedom))
    
    if num_links == 1 and not joint_details:
        print("\n단일 링크(고정 링크만 해당)이며 조인트가 없는 경우, 시스템의 상대적 자유도는 0입니다.")
        pass

    print("\n--- 계산 중 ---")
    dof = calculate_dof_grubler(num_links, joint_details, lambda_val)

    if dof is not None:
        print(f"\n==========================================")
        print(f"계산된 로봇 모델의 총 자유도 (DOF): {dof}")
        print(f"==========================================")

        if dof < 1 and num_links > 1 :
             if dof == 0:
                 print("INFO: 자유도가 0이므로, 이 메커니즘은 정적인 구조(structure)입니다.")
             elif dof < 0:
                 print("INFO: 자유도가 음수이므로, 이 메커니즘은 과도하게 구속된(overconstrained) 구조입니다.")
        elif dof ==1 and num_links > 1: # num_links > 1 조건 추가
            print("INFO: 자유도가 1이므로, 이 메커니즘은 단일 입력으로 제어 가능할 수 있습니다.")

        # 추가된 구성 분석 함수 호출
        if num_links > 0 : # 최소한 링크 1개는 있어야 분석 의미가 있음
             analyze_possible_configurations(num_links, joint_details, lambda_val, dof)

if __name__ == "__main__":
    get_user_input_and_calculate_dof()