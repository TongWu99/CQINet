'''
脚本功能：将array文件夹中的所有npy文件整合成一个npy文件
使用方法：
    方法1: python combine_csi_files.py
    方法2: python combine_csi_files.py --input_folder /path/to/your/array/folder
    方法3: 在脚本中直接修改 input_folder_path 变量
'''

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# 默认使用parameters.py（如果可用）
try:
    from parameters import *
    USE_PARAMETERS = True
except ImportError:
    USE_PARAMETERS = False
    print('警告: 无法导入parameters.py，请手动指定路径')

# ============ 配置区域：可以在这里直接修改路径 ============
# 手动指定array文件夹的父目录路径（如果指定了，将优先使用此路径）
# 格式1: 指向包含环境文件夹的目录，例如: "data/generated/generated_2_2_6_..."
# 格式2: 指向具体的array文件夹，例如: "data/generated/.../00032/array"
input_folder_path = None  # 设置为 None 表示使用parameters.py的路径或命令行参数

# 手动指定输出combined文件夹的位置（可选，默认保存在array文件夹同级的combined文件夹中）
output_folder_path = None  # 设置为 None 表示自动创建在array文件夹同级
# ==========================================================

def combine_csi_files(input_base_folder=None, output_base_folder=None):
    """
    整合array文件夹中的所有npy文件
    
    参数:
        input_base_folder: 输入文件夹路径（可选）
            - 如果指向包含环境文件夹的目录：会遍历所有环境
            - 如果指向具体的array文件夹：只处理该文件夹
        output_base_folder: 输出文件夹路径（可选，默认保存在array同级）
    
    输出：每个环境生成一个整合文件，保存在combined文件夹中
    """
    
    # 确定输入路径
    if input_base_folder is None:
        if input_folder_path is not None:
            input_base_folder = input_folder_path
        elif USE_PARAMETERS:
            input_base_folder = generatedFolder
        else:
            print('错误: 未指定输入路径！')
            print('请使用以下方式之一：')
            print('1. 在脚本中设置 input_folder_path 变量')
            print('2. 使用命令行参数: python combine_csi_files.py --input_folder /path/to/folder')
            print('3. 确保 parameters.py 中 generatedFolder 变量正确')
            return
    
    # 检查输入路径是否存在
    if not os.path.exists(input_base_folder):
        print(f'错误: 输入路径不存在: {input_base_folder}')
        return
    
    # 判断输入路径类型
    is_single_array_folder = os.path.basename(input_base_folder) == 'array'
    
    if is_single_array_folder:
        # 指向具体的array文件夹，只处理这一个
        array_folder = input_base_folder
        env_path = os.path.dirname(array_folder)
        env_name = os.path.basename(env_path)
        env_list = [(env_name, env_path)]
        print(f'处理单个array文件夹: {array_folder}')
    else:
        # 指向包含环境文件夹的目录，遍历所有环境
        env_list = []
        for item in os.listdir(input_base_folder):
            env_path = os.path.join(input_base_folder, item)
            if os.path.isdir(env_path):
                array_folder = os.path.join(env_path, 'array')
                if os.path.exists(array_folder):
                    env_list.append((item, env_path))
        env_list.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
        print(f'找到 {len(env_list)} 个包含array文件夹的环境')
        print(f'输入文件夹: {input_base_folder}')
    
    if len(env_list) == 0:
        print('警告: 未找到任何array文件夹')
        return
    
    # 处理每个环境
    for env_name, env_path in tqdm(env_list, desc='处理环境'):
        # 如果是单个array文件夹，已经设置好了，否则重新构建路径
        if not is_single_array_folder:
            array_folder = os.path.join(env_path, 'array')
        
        # 确定输出路径
        if output_base_folder is not None:
            combined_folder = os.path.join(output_base_folder, env_name, 'combined')
        elif output_folder_path is not None:
            combined_folder = os.path.join(output_folder_path, env_name, 'combined')
        else:
            combined_folder = os.path.join(env_path, 'combined')
        
        # 检查array文件夹是否存在
        if not os.path.exists(array_folder):
            print(f'警告: {array_folder} 不存在，跳过')
            continue
        
        # 创建combined文件夹
        if not os.path.exists(combined_folder):
            os.makedirs(combined_folder)
        
        # 自动检测BS和UE数量（如果parameters.py不可用或变量未定义）
        need_auto_detect = not USE_PARAMETERS
        if USE_PARAMETERS:
            # 检查必要的变量是否存在
            try:
                _ = BSnum
                _ = BSlist
                _ = UElist
                _ = scenario
            except NameError:
                # 变量未定义，需要自动检测
                need_auto_detect = True
                print('警告: parameters.py中缺少必要变量，将从文件自动检测')
        
        if need_auto_detect:
            # 从文件名推断BS和UE信息
            array_files = [f for f in os.listdir(array_folder) if f.endswith('.npy')]
            if len(array_files) == 0:
                print(f'警告: {array_folder} 中没有npy文件，跳过')
                continue
            
            # 解析文件名获取BS和UE列表
            bs_set = set()
            ue_set = set()
            detected_scenario = 2  # 默认假设scenario 2
            
            # 先检查前几个文件判断scenario类型
            for filename in array_files[:10]:
                if 'bs' in filename and 'ue' in filename:
                    parts = filename.replace('.npy', '').split('_')
                    if len(parts) == 2:
                        try:
                            ue_str = parts[1].replace('ue', '')
                            if len(ue_str) == 5:
                                detected_scenario = 2  # 5位数字表示scenario 2
                            else:
                                detected_scenario = 1  # 较短数字表示scenario 1
                            break
                        except:
                            pass
            
            # 遍历所有文件获取完整的BS和UE列表
            print(f'正在分析 {len(array_files)} 个文件...')
            for filename in array_files:
                # 格式: bs{bs}_ue{ue}.npy 或 bs{bs}_ue{ue:05d}.npy
                if 'bs' in filename and 'ue' in filename:
                    parts = filename.replace('.npy', '').split('_')
                    if len(parts) == 2:
                        try:
                            bs_set.add(int(parts[0].replace('bs', '')))
                            ue_str = parts[1].replace('ue', '')
                            ue_set.add(int(ue_str))
                        except:
                            pass
            
            BSlist = sorted(list(bs_set)) if bs_set else [0]
            UElist = sorted(list(ue_set)) if ue_set else []
            BSnum = len(BSlist)
            UEnum = len(UElist)
            scenario = detected_scenario
            print(f'自动检测: scenario={scenario}, BS={BSlist}, 实际找到的UE数量={UEnum}')
            
            # 如果是scenario 2，检查是否有缺失的文件
            if detected_scenario == 2 and UEnum < 10000:
                print(f'警告: 期望10000个UE文件，但只找到{UEnum}个文件')
                print(f'      这可能是正常的，如果数据生成未完成或UElist配置不完整')
        
        # 为每个BS处理
        for bs_idx in range(BSnum):
            bs_id = BSlist[bs_idx]
            
            # 方法1：直接读取文件夹中的所有文件（更准确）
            # 获取所有匹配的npy文件
            array_files = [f for f in os.listdir(array_folder) if f.endswith('.npy')]
            bs_files = [f for f in array_files if f.startswith(f'bs{bs_id}_ue')]
            
            if len(bs_files) == 0:
                print(f'警告: 环境 {env_name}, BS {bs_id} 没有找到CSI文件，跳过')
                continue
            
            # 按UE索引排序
            def get_ue_index(filename):
                try:
                    parts = filename.replace('.npy', '').split('_')
                    if len(parts) == 2:
                        return int(parts[1].replace('ue', ''))
                except:
                    return -1
                return -1
            
            bs_files.sort(key=get_ue_index)
            
            csi_list = []
            ue_indices = []
            
            # 加载所有找到的文件
            for filename in bs_files:
                file_path = os.path.join(array_folder, filename)
                try:
                    csi_data = np.load(file_path)  # 维度: (2, Nt*Nr, 子载波数)
                    csi_list.append(csi_data)
                    # 提取UE索引
                    ue_idx = get_ue_index(filename)
                    if ue_idx >= 0:
                        ue_indices.append(ue_idx)
                except Exception as e:
                    print(f'警告: 无法加载文件 {file_path}, 错误: {e}')
            
            if len(csi_list) == 0:
                print(f'警告: 环境 {env_name}, BS {bs_id} 没有成功加载任何CSI文件，跳过')
                continue
            
            # 检查是否有缺失的文件（如果使用parameters.py的UElist）
            if not need_auto_detect and USE_PARAMETERS and len(UElist) > 0:
                expected_count = len(UElist)
                actual_count = len(csi_list)
                if actual_count < expected_count:
                    missing_count = expected_count - actual_count
                    print(f'信息: 环境 {env_name}, BS {bs_id} 找到 {actual_count}/{expected_count} 个文件')
                    print(f'      缺失 {missing_count} 个文件（这是正常的，如果数据生成未完成）')
                elif actual_count == expected_count:
                    print(f'信息: 环境 {env_name}, BS {bs_id} 所有文件都已找到 ({actual_count} 个)')
            else:
                print(f'信息: 环境 {env_name}, BS {bs_id} 找到 {len(csi_list)} 个文件')
            
            # 整合所有CSI矩阵
            # csi_list中每个元素维度: (2, Nt*Nr, 子载波数)
            # 整合后维度: (2, UE数量, Nt*Nr, 子载波数)
            combined_array = np.stack(csi_list, axis=1)  # 在第1维（UE维度）堆叠
            print(f'\n环境 {env_name}, BS {bs_id}: 成功整合 {len(csi_list)}/{len(UElist)} 个用户的CSI')
            
            # 保存整合文件
            output_filename = f'BS{bs_id}_all_users.npy'
            output_path = os.path.join(combined_folder, output_filename)
            np.save(output_path, combined_array)
            
            print(f'  - 输出维度: {combined_array.shape}')
            print(f'  - 保存路径: {output_path}')
            
            # 显示文件大小
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f'  - 文件大小: {file_size_mb:.2f} MB')
    
    print('\n所有CSI文件整合完成!')


def verify_combined_file(file_path, expected_ue_count=None):
    """
    验证整合文件是否正确
    
    参数:
        file_path: 整合文件的路径
        expected_ue_count: 期望的用户数量（可选）
    """
    try:
        data = np.load(file_path)
        print(f'\n验证文件: {file_path}')
        print(f'  维度: {data.shape}')
        print(f'  数据类型: {data.dtype}')
        
        if len(data.shape) == 4:
            real_imag, ue_count, antenna_pairs, subcarriers = data.shape
            print(f'  实部/虚部: {real_imag}')
            print(f'  用户数量: {ue_count}')
            print(f'  天线对数量: {antenna_pairs}')
            print(f'  子载波数: {subcarriers}')
            
            if expected_ue_count and ue_count != expected_ue_count:
                print(f'  警告: 期望用户数 {expected_ue_count}，实际 {ue_count}')
            
            # 显示文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f'  文件大小: {file_size_mb:.2f} MB')
            
            # 重构复数矩阵示例
            csi_matrices = data[0] + 1j * data[1]
            print(f'  复数矩阵维度: {csi_matrices.shape}')
            print(f'  第一个用户CSI形状: {csi_matrices[0].shape}')
            
            return True
        else:
            print(f'  错误: 维度不正确，期望4维，实际{len(data.shape)}维')
            return False
            
    except Exception as e:
        print(f'  错误: 无法读取文件 - {e}')
        return False


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='整合CSI npy文件')
    parser.add_argument('--input_folder', type=str, default="/data/wutong/WAIRD/Wireless_AI_Research_Dataset/Dataset/data/RawData/generated_2_2_6_1000_1_8_4_1_1_1_74_52/01105/array",
                        help='输入文件夹路径（可指向包含环境文件夹的目录或具体的array文件夹）')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='输出文件夹路径（可选，默认保存在array文件夹同级）')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('CSI文件整合脚本')
    print('=' * 60)
    
    # 确定输入路径
    input_path = args.input_folder if args.input_folder else input_folder_path
    
    if USE_PARAMETERS:
        print(f'场景: scenario_{scenario}')
        print(f'BS列表: {BSlist}')
        print(f'UE列表: 前10个 {UElist[:10]}, 总共 {len(UElist)} 个')
        if input_path is None:
            print(f'生成文件夹: {generatedFolder}')
    else:
        print('未使用parameters.py，将从文件自动检测配置')
        if input_path:
            print(f'指定输入路径: {input_path}')
        else:
            print('错误: 请指定输入路径！')
            print('使用方法:')
            print('  1. 在脚本中设置 input_folder_path 变量')
            print('  2. 使用命令行: python combine_csi_files.py --input_folder /path/to/folder')
            sys.exit(1)
    
    print('=' * 60)
    print()
    
    # 执行整合
    combine_csi_files(input_base_folder=input_path, output_base_folder=args.output_folder)
    
    # 验证第一个整合文件（如果使用了parameters.py）
    if USE_PARAMETERS and input_path is None:
        if os.path.exists(generatedFolder):
            env_list = os.listdir(generatedFolder)
            env_list.sort(key=lambda x: int(x) if x.isdigit() else 0)
            if len(env_list) > 0:
                first_env = env_list[0]
                first_bs = BSlist[0]
                verify_path = os.path.join(
                    generatedFolder, 
                    first_env, 
                    'combined', 
                    f'BS{first_bs}_all_users.npy'
                )
                if os.path.exists(verify_path):
                    print('\n' + '=' * 60)
                    print('验证示例文件:')
                    verify_combined_file(verify_path, expected_ue_count=len(UElist))
