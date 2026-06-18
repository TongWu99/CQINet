'''
将.npy文件转换为.mat文件（供MATLAB使用）
功能：将com.py生成的合并CSI文件转换为MATLAB可读的.mat格式
     同时封装场景信息（位置信息、环境俯视图等）

使用方法：
    python convert_npy_to_mat.py --input_file path/to/BS0_all_users.npy
    或
    python convert_npy_to_mat.py --input_folder path/to/combined/
'''

import os
import sys
import argparse
import numpy as np
import scipy.io
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 尝试导入parameters.py
try:
    from parameters import *
    USE_PARAMETERS = True
except ImportError:
    USE_PARAMETERS = False
    print('提示: 无法导入parameters.py，将使用默认值或从路径推断')


def find_scenario_files(csi_file_path):
    """
    根据CSI文件路径自动查找对应的场景文件
    
    参数:
        csi_file_path: CSI文件路径（如 .../00032/combined/BS0_all_users.npy）
    
    返回:
        info_file: Info.npy文件路径
        env_image_file: environment.png文件路径
        env_id: 环境ID（如 '00032'）
    """
    # 获取CSI文件的目录
    csi_dir = os.path.dirname(os.path.abspath(csi_file_path))
    
    # 尝试从路径中提取环境ID
    env_id = None
    path_parts = csi_file_path.replace('\\', '/').split('/')
    for part in path_parts:
        if len(part) == 5 and part.isdigit():
            env_id = part
            break
    
    # 查找Info.npy和environment.png
    info_file = None
    env_image_file = None
    
    # 方法1: 在同级目录的父目录中查找（combined/../Info.npy）
    parent_dir = os.path.dirname(csi_dir)
    potential_info = os.path.join(parent_dir, 'Info.npy')
    potential_env = os.path.join(parent_dir, 'environment.png')
    
    if os.path.exists(potential_info):
        info_file = potential_info
    if os.path.exists(potential_env):
        env_image_file = potential_env
    
    # 方法2: 在scenario文件夹中查找
    if not info_file or not env_image_file:
        for part in path_parts:
            if 'scenario' in part.lower() and env_id:
                # 构建scenario路径
                scenario_idx = path_parts.index(part)
                scenario_path = '/'.join(path_parts[:scenario_idx+1])
                potential_info = os.path.join(scenario_path, env_id, 'Info.npy')
                potential_env = os.path.join(scenario_path, env_id, 'environment.png')
                
                if os.path.exists(potential_info) and not info_file:
                    info_file = potential_info
                if os.path.exists(potential_env) and not env_image_file:
                    env_image_file = potential_env
                break
    
    return info_file, env_image_file, env_id


def load_info_npy(info_file, scenario=2):
    """
    加载Info.npy文件并解析
    
    参数:
        info_file: Info.npy文件路径
        scenario: 场景编号（1或2）
    
    返回:
        info_dict: 包含解析后信息的字典
    """
    if not os.path.exists(info_file):
        return None
    
    try:
        info_data = np.load(info_file, allow_pickle=True, encoding='latin1')
        
        info_dict = {
            'raw_data': info_data,
            'scenario': scenario
        }
        
        if scenario == 2:
            # Scenario 2格式
            if len(info_data) >= 4:
                info_dict['img_size'] = np.floor(info_data[:2]).astype(int).tolist()
                info_dict['bs_loc'] = info_data[2:4].tolist()
                
                if len(info_data) > 4:
                    ue_data = info_data[4:]
                    num_ue = len(ue_data) // 2
                    info_dict['ue_locs'] = ue_data.reshape(-1, 2).tolist()
                    info_dict['num_ue'] = num_ue
        else:
            # Scenario 1格式
            if len(info_data) >= 2:
                info_dict['img_size'] = np.floor(info_data[:2]).astype(int).tolist()
                if len(info_data) > 2:
                    link_data = info_data[2:]
                    num_links = len(link_data) // 4
                    info_dict['link_matrix'] = link_data.reshape(-1, 4).tolist()
                    info_dict['num_links'] = num_links
        
        return info_dict
        
    except Exception as e:
        print(f'警告: 无法加载Info.npy文件: {e}')
        return None


def load_environment_image(env_image_file):
    """
    加载环境俯视图PNG文件
    
    参数:
        env_image_file: environment.png文件路径
    
    返回:
        image_array: 图像数组 (height, width, channels) 或 (height, width) for grayscale
        image_info: 图像信息字典
    """
    if not os.path.exists(env_image_file):
        return None, None
    
    try:
        img = Image.open(env_image_file)
        img_array = np.array(img)
        
        image_info = {
            'shape': img_array.shape,
            'mode': img.mode,
            'size': img.size,  # (width, height)
            'format': img.format
        }
        
        return img_array, image_info
        
    except Exception as e:
        print(f'警告: 无法加载环境图像: {e}')
        return None, None


def create_annotated_map(env_image_array, info_dict, output_path, scenario=2, 
                         max_ue_display=5000, bs_marker_size=15, ue_marker_size=2):
    """
    创建标注了基站和用户位置的2D俯视图
    
    参数:
        env_image_array: 环境图像数组
        info_dict: 位置信息字典（从load_info_npy返回）
        output_path: 输出图像路径
        scenario: 场景编号（1或2）
        max_ue_display: 最大显示的UE数量（如果用户太多，进行采样）
        bs_marker_size: 基站标记大小
        ue_marker_size: 用户标记大小
    """
    if env_image_array is None or info_dict is None:
        print('  警告: 缺少环境图像或位置信息，无法生成标注图')
        return False
    
    try:
        # 获取图像尺寸（使用实际图像尺寸，不再resize）
        # 优先使用实际图像尺寸，而不是Info.npy中的尺寸
        if len(env_image_array.shape) == 3:
            img_height, img_width = env_image_array.shape[:2]
        else:
            img_height, img_width = env_image_array.shape
        
        # 如果Info.npy中有img_size，用于坐标缩放（但不resize图像）
        info_img_width = None
        info_img_height = None
        if 'img_size' in info_dict:
            info_img_width, info_img_height = info_dict['img_size']
            info_img_width, info_img_height = int(info_img_width), int(info_img_height)
            
            # 如果尺寸不一致，提示并计算缩放比例
            if info_img_width != img_width or info_img_height != img_height:
                print(f'  提示: 图像尺寸与Info.npy不一致')
                print(f'    Info.npy指定: [{info_img_width}, {info_img_height}] (W×H)')
                print(f'    实际图像: [{img_width}, {img_height}] (W×H)')
                print(f'    坐标将按比例缩放以匹配实际图像尺寸')
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 显示环境图像
        # 保持图像原始方向（y轴从上到下），只反转坐标
        if len(env_image_array.shape) == 3:
            # RGB或RGBA图像
            if env_image_array.shape[2] == 4:
                # RGBA图像，需要处理alpha通道
                ax.imshow(env_image_array, extent=[0, img_width, img_height, 0], 
                         origin='upper', alpha=0.8)
            else:
                # RGB图像
                ax.imshow(env_image_array, extent=[0, img_width, img_height, 0], 
                         origin='upper', alpha=0.8)
        else:
            # 灰度图像
            ax.imshow(env_image_array, extent=[0, img_width, img_height, 0], 
                     origin='upper', cmap='gray', alpha=0.8)
        
        # 绘制基站位置
        if 'bs_loc' in info_dict and info_dict['bs_loc'] is not None:
            bs_loc = info_dict['bs_loc']
            if len(bs_loc) >= 2:
                # Info.npy中的坐标通常是实际像素坐标（基于Info.npy中的img_size）
                # 如果实际图像尺寸与Info.npy中的img_size不一致，需要按比例缩放坐标
                if info_img_width is not None and info_img_height is not None:
                    # 计算缩放比例
                    scale_x = img_width / info_img_width if info_img_width > 0 else 1
                    scale_y = img_height / info_img_height if info_img_height > 0 else 1
                else:
                    scale_x = 1
                    scale_y = 1
                
                if bs_loc[0] > info_img_width * 1.5 if info_img_width else img_width * 1.5 or \
                   bs_loc[1] > info_img_height * 1.5 if info_img_height else img_height * 1.5:
                    # 可能是归一化坐标，需要映射
                    if bs_loc[0] <= 1 and bs_loc[1] <= 1:
                        # 归一化坐标，先映射到Info.npy的img_size，再缩放到实际图像尺寸
                        bs_x = bs_loc[0] * (info_img_width if info_img_width else img_width) * scale_x
                        bs_y = bs_loc[1] * (info_img_height if info_img_height else img_height) * scale_y
                    else:
                        # 坐标超出范围，可能是归一化但值>1，或需要按比例缩放
                        # 假设是归一化坐标
                        bs_x = bs_loc[0] * (info_img_width if info_img_width else img_width) * scale_x
                        bs_y = bs_loc[1] * (info_img_height if info_img_height else img_height) * scale_y
                else:
                    # 实际像素坐标（基于Info.npy的img_size），需要缩放到实际图像尺寸
                    bs_x = bs_loc[0] * scale_x
                    bs_y = bs_loc[1] * scale_y
                
                # 转换y坐标：从图像坐标系（上到下）转换为数学坐标系（下到上）
                bs_y = img_height - bs_y
                
                # 绘制基站（红色五角星）
                ax.scatter(bs_x, bs_y, s=bs_marker_size**2, marker='*', 
                          c='red', edgecolors='darkred', linewidths=2,
                          label='Base Station', zorder=10)
                # 添加文本标注
                ax.annotate('BS', (bs_x, bs_y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12,
                           color='red', weight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.7, edgecolor='red'))
        
        # 绘制用户位置
        if 'ue_locs' in info_dict and info_dict['ue_locs'] is not None:
            ue_locs = np.array(info_dict['ue_locs'])
            num_ue = len(ue_locs)
            
            if num_ue > 0:
                # Info.npy中的UE坐标通常是实际像素坐标（基于Info.npy中的img_size）
                # 如果实际图像尺寸与Info.npy中的img_size不一致，需要按比例缩放坐标
                if info_img_width is not None and info_img_height is not None:
                    # 计算缩放比例
                    scale_x = img_width / info_img_width if info_img_width > 0 else 1
                    scale_y = img_height / info_img_height if info_img_height > 0 else 1
                else:
                    scale_x = 1
                    scale_y = 1
                
                # 检查坐标范围以判断是否需要缩放
                max_x, max_y = np.max(ue_locs[:, 0]), np.max(ue_locs[:, 1])
                min_x, min_y = np.min(ue_locs[:, 0]), np.min(ue_locs[:, 1])
                
                # 如果所有坐标都在[0,1]范围内，认为是归一化坐标
                if max_x <= 1 and max_y <= 1 and min_x >= 0 and min_y >= 0:
                    # 归一化坐标，先映射到Info.npy的img_size，再缩放到实际图像尺寸
                    ref_width = info_img_width if info_img_width else img_width
                    ref_height = info_img_height if info_img_height else img_height
                    ue_x = ue_locs[:, 0] * ref_width * scale_x
                    ue_y = ue_locs[:, 1] * ref_height * scale_y
                elif max_x > (info_img_width if info_img_width else img_width) * 1.5 or \
                     max_y > (info_img_height if info_img_height else img_height) * 1.5:
                    # 坐标超出范围，可能是归一化但值>1，尝试按比例缩放
                    print(f'  警告: UE坐标范围异常，尝试自动调整')
                    print(f'    X范围: [{min_x:.2f}, {max_x:.2f}], 参考宽度: {info_img_width if info_img_width else img_width}')
                    print(f'    Y范围: [{min_y:.2f}, {max_y:.2f}], 参考高度: {info_img_height if info_img_height else img_height}')
                    # 尝试按比例缩放
                    ref_width = info_img_width if info_img_width else img_width
                    ref_height = info_img_height if info_img_height else img_height
                    scale_x_auto = img_width / max_x if max_x > 0 else 1
                    scale_y_auto = img_height / max_y if max_y > 0 else 1
                    ue_x = ue_locs[:, 0] * scale_x_auto
                    ue_y = ue_locs[:, 1] * scale_y_auto
                else:
                    # 实际像素坐标（基于Info.npy的img_size），需要缩放到实际图像尺寸
                    ue_x = ue_locs[:, 0] * scale_x
                    ue_y = ue_locs[:, 1] * scale_y
                
                # 转换y坐标：从图像坐标系（上到下）转换为数学坐标系（下到上）
                ue_y = img_height - ue_y
                
                # 如果用户太多，进行采样
                if num_ue > max_ue_display:
                    indices = np.random.choice(num_ue, max_ue_display, replace=False)
                    ue_x_display = ue_x[indices]
                    ue_y_display = ue_y[indices]
                    print(f'  提示: UE数量({num_ue})过多，随机采样{max_ue_display}个进行显示')
                else:
                    ue_x_display = ue_x
                    ue_y_display = ue_y
                
                # 绘制用户（蓝色小点）
                ax.scatter(ue_x_display, ue_y_display, s=ue_marker_size**2, 
                          c='cyan', alpha=0.6, edgecolors='blue', 
                          linewidths=0.3, label=f'User Equipment ({num_ue})', zorder=5)
                for idx in range(0, num_ue, 100):

                    ax.annotate(
                        str(idx),
                        (ue_x[idx], ue_y[idx]),
                        fontsize=6,
                        color='red'
                    )
        
        # 设置图形属性
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # y轴从上到下（0在顶部，img_height在底部），与图像方向一致
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
        ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
        
        # 添加标题
        env_id = info_dict.get('env_id', 'Unknown')
        title = f'Environment Top View with BS and UE Locations (Env: {env_id})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 添加网格（可选）
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f'  ✓ 已生成标注图: {output_path}')
        return True
        
    except Exception as e:
        print(f'  错误: 生成标注图失败 - {e}')
        import traceback
        traceback.print_exc()
        return False


def convert_npy_to_mat(npy_file, mat_file=None, variable_name='csi_data',
                       info_file=None, env_image_file=None, scenario=2,
                       include_scenario_info=True, generate_annotated_map=True):
    """
    将.npy文件转换为.mat文件，并封装场景信息
    
    参数:
        npy_file: 输入的.npy文件路径
        mat_file: 输出的.mat文件路径（可选，默认与.npy同目录同名）
        variable_name: MATLAB变量名（默认'csi_data'）
        info_file: Info.npy文件路径（可选，会自动查找）
        env_image_file: environment.png文件路径（可选，会自动查找）
        scenario: 场景编号（1或2）
        include_scenario_info: 是否包含场景信息
    """
    if not os.path.exists(npy_file):
        print(f'错误: 文件不存在: {npy_file}')
        return False
    
    # 确定输出文件路径
    if mat_file is None:
        mat_file = os.path.splitext(npy_file)[0] + '.mat'
    
    try:
        # 加载CSI数据
        print(f'\n读取CSI文件: {npy_file}')
        csi_data = np.load(npy_file)
        print(f'  CSI维度: {csi_data.shape}')
        print(f'  CSI数据类型: {csi_data.dtype}')
        
        # 准备保存的数据字典
        mat_data = {variable_name: csi_data}
        
        # 加载场景信息
        if include_scenario_info:
            print(f'\n加载场景信息...')
            
            # 自动查找场景文件
            if not info_file or not env_image_file:
                auto_info, auto_env, env_id = find_scenario_files(npy_file)
                if not info_file:
                    info_file = auto_info
                if not env_image_file:
                    env_image_file = auto_env
                if env_id:
                    print(f'  检测到环境ID: {env_id}')
            
            # 加载Info.npy
            info_dict = None
            if info_file:
                if os.path.exists(info_file):
                    print(f'  读取位置信息: {info_file}')
                    info_dict = load_info_npy(info_file, scenario)
                    if info_dict:
                        print(f'    ✓ 图像尺寸: {info_dict.get("img_size", "N/A")}')
                        print(f'    ✓ BS位置: {info_dict.get("bs_loc", "N/A")}')
                        if 'ue_locs' in info_dict:
                            print(f'    ✓ UE数量: {info_dict.get("num_ue", "N/A")}')
                else:
                    print(f'  警告: Info.npy文件不存在: {info_file}')
            else:
                print(f'  提示: 未指定Info.npy文件路径，将跳过位置信息')
            
            # 加载环境图像
            env_image_array = None
            env_image_info = None
            if env_image_file:
                if os.path.exists(env_image_file):
                    print(f'  读取环境图像: {env_image_file}')
                    env_image_array, env_image_info = load_environment_image(env_image_file)
                    if env_image_array is not None:
                        original_size = env_image_info["size"]  # (width, height)
                        print(f'    ✓ 原始图像尺寸: {original_size} (W×H)')
                        print(f'    ✓ 图像模式: {env_image_info["mode"]}')
                        print(f'    ✓ 数组形状: {env_image_array.shape}')
                        
                        # 不再根据Info.npy中的图像尺寸调整环境图像，保持原始尺寸
                        # 注意：如果图像尺寸与Info.npy中的img_size不一致，坐标可能需要相应调整
                        if info_dict and 'img_size' in info_dict:
                            target_size = info_dict['img_size']  # [width, height]
                            target_width, target_height = int(target_size[0]), int(target_size[1])
                            
                            # 仅提示尺寸差异，不进行resize
                            if original_size[0] != target_width or original_size[1] != target_height:
                                print(f'    ⚠️  图像尺寸与Info.npy不一致（但不进行resize）')
                                print(f'       Info.npy指定尺寸: [{target_width}, {target_height}] (W×H)')
                                print(f'       实际图像尺寸: {original_size} (W×H)')
                                print(f'       注意：坐标系统将使用实际图像尺寸')
                            
                        env_image_info['resized'] = False
                        env_image_info['original_size'] = original_size
                else:
                    print(f'  警告: 环境图像文件不存在: {env_image_file}')
            else:
                print(f'  提示: 未指定环境图像文件路径，将跳过图像')
            
            # 创建场景信息结构
            scenario_info = {}
            
            # 添加位置信息
            if info_dict:
                scenario_info['info_data'] = info_dict['raw_data']
                scenario_info['scenario'] = scenario
                if 'img_size' in info_dict:
                    scenario_info['img_size'] = np.array(info_dict['img_size'])
                if 'bs_loc' in info_dict:
                    scenario_info['bs_loc'] = np.array(info_dict['bs_loc'])
                if 'ue_locs' in info_dict:
                    scenario_info['ue_locs'] = np.array(info_dict['ue_locs'])
                if 'num_ue' in info_dict:
                    scenario_info['num_ue'] = info_dict['num_ue']
            
            # 添加环境图像
            if env_image_array is not None:
                # 保存图像数组（MATLAB可以直接读取）
                scenario_info['environment_image'] = env_image_array
                scenario_info['environment_image_info'] = env_image_info
            
            # 从文件路径提取元数据
            path_parts = npy_file.replace('\\', '/').split('/')
            for part in path_parts:
                if len(part) == 5 and part.isdigit():
                    scenario_info['env_id'] = part
                    break
            
            # 尝试从路径提取载波频率
            for part in path_parts:
                if '_2_6_' in part or part.endswith('_2_6'):
                    scenario_info['carrier_freq'] = '2_6'
                    break
                elif '_28_0_' in part or part.endswith('_28_0'):
                    scenario_info['carrier_freq'] = '28_0'
                    break
                elif '_60_0_' in part or part.endswith('_60_0'):
                    scenario_info['carrier_freq'] = '60_0'
                    break
                elif '_100_0_' in part or part.endswith('_100_0'):
                    scenario_info['carrier_freq'] = '100_0'
                    break
            
            # 添加系统参数（如果可用）
            if USE_PARAMETERS:
                try:
                    scenario_info['Nt'] = np.array(Nt)
                    scenario_info['Nr'] = np.array(Nr)
                    scenario_info['sampledCarriers'] = sampledCarriers
                    scenario_info['BWGHz'] = BWGHz
                except:
                    pass
            
            # 将场景信息添加到mat_data
            if scenario_info:
                mat_data['scenario_info'] = scenario_info
                print(f'\n  ✓ 场景信息已封装')
                print(f'    包含字段: {list(scenario_info.keys())}')
        
        # 保存为.mat文件
        print(f'\n保存MAT文件: {mat_file}')
        scipy.io.savemat(mat_file, mat_data, do_compression=True)
        
        # 显示文件大小
        npy_size = os.path.getsize(npy_file) / (1024 * 1024)
        mat_size = os.path.getsize(mat_file) / (1024 * 1024)
        print(f'  原始文件大小: {npy_size:.2f} MB')
        print(f'  转换后大小: {mat_size:.2f} MB')
        print(f'  MATLAB变量: {list(mat_data.keys())}')
        
        # 生成标注图（如果有场景信息）
        if generate_annotated_map and include_scenario_info and info_dict and env_image_array is not None:
            print(f'\n生成标注图...')
            annotated_map_path = os.path.splitext(mat_file)[0] + '_annotated.png'
            
            # 添加env_id到info_dict（如果还没有）
            if 'env_id' not in info_dict:
                path_parts = npy_file.replace('\\', '/').split('/')
                for part in path_parts:
                    if len(part) == 5 and part.isdigit():
                        info_dict['env_id'] = part
                        break
            
            create_annotated_map(env_image_array, info_dict, annotated_map_path, 
                               scenario=scenario, max_ue_display=5000)
        
        return True
        
    except Exception as e:
        print(f'错误: 转换失败 - {e}')
        import traceback
        traceback.print_exc()
        return False


def convert_folder(input_folder, output_folder=None, variable_name='csi_data',
                   info_folder=None, env_image_folder=None, scenario=2,
                   include_scenario_info=True):
    """
    批量转换文件夹中的所有.npy文件
    
    参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径（可选，默认与输入文件夹相同）
        variable_name: MATLAB变量名
        info_folder: Info.npy文件所在文件夹（可选，会自动查找）
        env_image_folder: environment.png文件所在文件夹（可选，会自动查找）
        scenario: 场景编号
        include_scenario_info: 是否包含场景信息
    """
    if not os.path.exists(input_folder):
        print(f'错误: 文件夹不存在: {input_folder}')
        return
    
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有.npy文件
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        print(f'警告: 在 {input_folder} 中未找到.npy文件')
        return
    
    print(f'找到 {len(npy_files)} 个.npy文件')
    
    success_count = 0
    for npy_file in tqdm(npy_files, desc='转换文件'):
        input_path = os.path.join(input_folder, npy_file)
        output_path = os.path.join(output_folder, os.path.splitext(npy_file)[0] + '.mat')
        
        if convert_npy_to_mat(input_path, output_path, variable_name,
                             info_file=None, env_image_file=None, scenario=scenario,
                             include_scenario_info=include_scenario_info,
                             generate_annotated_map=True):
            success_count += 1
        print()  # 空行分隔
    
    print(f'\n转换完成: {success_count}/{len(npy_files)} 个文件成功')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='将.npy文件转换为.mat文件（包含场景信息）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 转换单个文件（自动查找场景信息）
  python convert_npy_to_mat.py --input_file path/to/BS0_all_users.npy
  
  # 指定场景信息文件路径
  python convert_npy_to_mat.py --input_file path/to/BS0_all_users.npy \\
      --info_file path/to/Info.npy --env_image path/to/environment.png
  
  # 批量转换
  python convert_npy_to_mat.py --input_folder path/to/combined/
        '''
    )
    parser.add_argument('--input_file', type=str, default="/data/wutong/WAIRD/Wireless_AI_Research_Dataset/Dataset/data/RawData/generated_2_2_6_1000_1_8_4_1_1_1_9_52/00390/combined/BS0_all_users.npy",
                        help='输入的.npy文件路径')
    parser.add_argument('--input_folder', type=str, default=None,
                        help='输入文件夹路径（批量转换）')
    parser.add_argument('--output_file', type=str, default="/data/wutong/WAIRD/Wireless_AI_Research_Dataset/Dataset/data/RawData/generated_2_2_6_1000_1_8_4_1_1_1_9_52/00390/combined/00390.mat",
                        help='输出的.mat文件路径（可选）')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='输出文件夹路径（批量转换时使用）')
    parser.add_argument('--variable_name', type=str, default='csi_data',
                        help='MATLAB变量名（默认: csi_data）')
    parser.add_argument('--info_file', type=str, default="/data/wutong/WAIRD/Wireless_AI_Research_Dataset/Dataset/data/scenario_2/00390/Info.npy",
                        help='Info.npy文件路径（可选，会自动查找）')
    parser.add_argument('--env_image', type=str, default="/data/wutong/WAIRD/Wireless_AI_Research_Dataset/Dataset/data/scenario_2/00390/environment.png",
                        help='environment.png文件路径（可选，会自动查找）')
    parser.add_argument('--scenario', type=int, default=2,
                        help='场景编号 (1或2)，默认2')
    parser.add_argument('--no_scenario_info', action='store_true',
                        help='不包含场景信息（仅转换CSI数据）')
    parser.add_argument('--no_annotated_map', action='store_true',
                        help='不生成标注图')#ax.scatter
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('NPY to MAT 转换工具（含场景信息封装）')
    print('=' * 60)
    
    include_scenario = not args.no_scenario_info
    
    generate_map = not args.no_annotated_map
    
    if args.input_file:
        # 转换单个文件
        convert_npy_to_mat(
            args.input_file, 
            args.output_file, 
            args.variable_name,
            info_file=args.info_file,
            env_image_file=args.env_image,
            scenario=args.scenario,
            include_scenario_info=include_scenario,
            generate_annotated_map=generate_map
        )
    elif args.input_folder:
        # 批量转换
        convert_folder(
            args.input_folder, 
            args.output_folder, 
            args.variable_name,
            info_folder=args.info_file,  # 可以传入文件夹路径
            env_image_folder=args.env_image,  # 可以传入文件夹路径
            scenario=args.scenario,
            include_scenario_info=include_scenario,
            generate_annotated_map=generate_map
        )
    else:
        print('错误: 请指定 --input_file 或 --input_folder')
        parser.print_help()
#python NpyToMat.py