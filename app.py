#2025年度交通流统计分析系统

import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

class TrafficFlowAnalyzer:
    #交通流分析器
    def __init__(self, config_path="config.yaml"):

        self.config=self._load_config(config_path)
        # 哨兵值（用于替换原来的 NaN 或异常值）
        self.nan_sentinel = self.config.get('processing', {}).get('nan_sentinel', -1)
        self.device=self._setup_device()
        self.ais_data=None
        self.ship_info=None
    def _load_config(self, config_path)->Dict:
        #加载配置文件
        with open(config_path, 'r', encoding='utf-8') as file:
            config=yaml.safe_load(file)
        return config

    def _setup_device(self):
        #设置计算设备
        if torch.cuda.is_available() and self.config.get('use_cuda', False):
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    def load_data(self):
        #加载预处理后的AIS数据
        ais_ship_data_path=self.config['data_files']['premerged_ais']
        self.ais_data=pd.read_csv(ais_ship_data_path)
        print(f"Loaded AIS data with {len(self.ais_data)} records.")
        return self.ais_data
    #船型配置读取，预备最后的遍历船型分析
    def ship_type_filter(self,config):
        ship_types=config.get('ship_types', [])
        return ship_types
    #读取配置文件中的待分析截面列表
    def section_list_loader(self, config):
        """
        从配置文件加载截面定义
                - export_trajectories: 是否导出轨迹--还没做
        """
        raw_sections = config.get('cross_sections', [])
        
        sections = []
        for section in raw_sections:
            processed_section = {
                'id': section.get('id'),
                'name': section.get('name'),
                'point1_lon': section['point1']['lon'],
                'point1_lat': section['point1']['lat'],
                'point2_lon': section['point2']['lon'],
                'point2_lat': section['point2']['lat'],
                'export_trajectories': section.get('export_trajectories', False)
            }
            sections.append(processed_section)
        
        print(f"加载了 {len(sections)} 条截面定义")
        return sections 
    def tensor_builder(self,ais_data):
        #np化AIS数据，再张量化
        print("    [tensor_builder] 原始数据列数:", ais_data.shape[1])
        
        # 处理时间戳列 - 转换为Unix时间戳（秒）
        ais_data_copy = ais_data.copy()
        ais_data_copy['time'] = pd.to_datetime(ais_data_copy.iloc[:, 0])
        '''如果是ais_data_copy['time']. astype('int64') / 1e9会引发时间指数级错误'''
        ais_data_copy['time'] = ais_data_copy['time'].apply(lambda x: x.timestamp())
        
        # 选择需要的列：time, mmsi, lon, lat, type, length, width
        # 原始列顺序：time(0), mmsi(1), status(2), lon(3), lat(4), ..., type(10), length(11), width(12)
        #selected_cols = ['time', ais_data.columns[1], ais_data.columns[3], ais_data.columns[4], 
        #                ais_data.columns[10], ais_data.columns[11], ais_data.columns[12]]
        selected_cols=['time','mmsi','lon','lat','type','length']
        ais_data_selected = ais_data_copy[selected_cols].copy()
        
        # 转换为numpy数组
        nps = ais_data_selected.to_numpy()
        del ais_data, ais_data_copy, ais_data_selected
        
        # 逐列创建张量并立即添加到列表，避免同时保留多个中间张量
        tensors = []
        
        for col_idx in range(nps.shape[1]):
            col_data = nps[:, col_idx]
            
            # 转换为数字（保留为 Series 以便后续处理）
            try:
                # pd.to_numeric 在输入 numpy 数组时会返回 ndarray；强制包装为 pd.Series 以确保有 fillna/mask 方法
                col_series = pd.Series(pd.to_numeric(col_data, errors='coerce'))  # 强制转换为数字，错误值变NaN，确保为 Series
            except:
                col_series = pd.Series(np.array(col_data, dtype='float64'))
            
            # 对特定列应用哨兵规则
            # 对 type：把 >100 的异常值也视为未知并替换为哨兵
            if selected_cols[col_idx] == 'type':
                col_series = col_series.mask(col_series > 100, self.nan_sentinel)
            # 全部 NaN 替换为哨兵（统一口径）
            col_data_filled = col_series.fillna(self.nan_sentinel).values

            # 创建张量（使用 float32）
            tensor = torch.tensor(col_data_filled, dtype=torch.float32, device=self.device)

            tensors.append(tensor)
        
        del nps
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        ais_tensor = torch.stack(tensors, dim=1)
        del tensors
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        print(f"    [tensor_builder] 张量形状: {ais_tensor.shape}")
        ais_tensor = ais_tensor.to(self.device)
        return ais_tensor
    def track_builder(self,ais_tensor):
        #先按时间排序再按mmsi排序，构建完整轨迹
        print("    [track_builder] 排序前张量形状:", ais_tensor.shape)
        
        # 先按MMSI排序
        indices = torch.argsort(ais_tensor[:, 1], stable=True)
        sorted_by_mmsi = ais_tensor[indices]
        del ais_tensor, indices
        
        # 再在每个MMSI组内按时间排序
        indices = torch.argsort(sorted_by_mmsi[:, 0], stable=True)
        sorted_ais = sorted_by_mmsi[indices]
        del sorted_by_mmsi, indices
        
        print("    [track_builder] 排序后张量形状:", sorted_ais.shape)
        return sorted_ais
    def get_route_segments(self, final_sorted_nps_tensor, time_col=0, mmsi_col=1, lon_col=2, lat_col=3, type_col=4, length_col=5, width_col=6):
        print("    [get_route_segments] 输入张量形状:", final_sorted_nps_tensor.shape)
        
        # 步骤1：提取mmsi生成掩码，用于判断相邻记录是否属于同一船舶
        mmsi = final_sorted_nps_tensor[:, mmsi_col]
        same_mmsi_mask = (mmsi[1:] == mmsi[:-1])
        print(f"    [get_route_segments] 相同MMSI的相邻记录数: {same_mmsi_mask.sum().item()}")
        
        mmsi_filtered = mmsi[1:][same_mmsi_mask]
        del mmsi
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤2：处理经度
        lon = final_sorted_nps_tensor[:, lon_col]
        start_lon = lon[:-1][same_mmsi_mask]
        end_lon = lon[1:][same_mmsi_mask]
        del lon
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤3：处理纬度
        lat = final_sorted_nps_tensor[:, lat_col]
        start_lat = lat[:-1][same_mmsi_mask]
        end_lat = lat[1:][same_mmsi_mask]
        del lat
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤4：处理时间
        time = final_sorted_nps_tensor[:, time_col]
        time_filtered = time[1:][same_mmsi_mask]
        del time
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤5：处理船型
        type_data = final_sorted_nps_tensor[:, type_col]
        type_filtered = type_data[1:][same_mmsi_mask]
        del type_data
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤6：处理长度
        length = final_sorted_nps_tensor[:, length_col]
        length_filtered = length[1:][same_mmsi_mask]
        del length
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        # 步骤7：处理宽度，并删除掩码  目前不需要宽度参数，已放弃
        ''' width = final_sorted_nps_tensor[:, width_col]
        width_filtered = width[1:][same_mmsi_mask]
        del width, same_mmsi_mask
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None'''
        
        # 步骤8：最后才堆叠张量
        # 输出列顺序：0=mmsi, 1=start_lon, 2=start_lat, 3=end_lon, 4=end_lat, 5=time, 6=type, 7=length, 8=width
        segments = torch.stack([mmsi_filtered, start_lon, start_lat, end_lon, end_lat, 
                               time_filtered, type_filtered, length_filtered], dim=1)
        del mmsi_filtered, start_lon, start_lat, end_lon, end_lat, time_filtered, type_filtered, length_filtered
        torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        return segments
    #相交检测算法
    def line_intersect_calculate(self,segments, point1_lon, point1_lat, point2_lon, point2_lat):
        def cross(x1, y1, x2, y2, x3, y3):
            return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        # segments columns: 0=mmsi, 1=start_lon, 2=start_lat, 3=end_lon, 4=end_lat, ...
        s1_x, s1_y, s2_x, s2_y = segments[:, 1], segments[:, 2], segments[:, 3], segments[:, 4]
        d1 = cross(s1_x, s1_y, s2_x, s2_y, point1_lon, point1_lat)
        d2 = cross(s1_x, s1_y, s2_x, s2_y, point2_lon, point2_lat)
        d3 = cross(point1_lon, point1_lat, point2_lon, point2_lat, s1_x, s1_y)
        d4 = cross(point1_lon, point1_lat, point2_lon, point2_lat, s2_x, s2_y)
        intersect_mask = ((d1 * d2) < 0) & ((d3 * d4) < 0)
        intersected_numbers = intersect_mask.sum().item()
        intersect_lines = segments[intersect_mask]
        return intersected_numbers, intersect_lines
    def get_ship_type_mapping(self, config: Dict) -> Dict[str, List[int]]:
        """获取船型名称到类型代码的映射"""
        return config.get('ship_types', {})
    
    def get_length_categories(self, config: Dict) -> List[Dict]:
        """获取船长分类标准"""
        return config.get('length_categories', [])
    
    def filter_by_type_codes(self, intersect_lines: torch.Tensor, type_codes: List[int], type_col: int = 6) -> torch.Tensor:
        """按类型代码筛选线段"""
        type_codes_tensor = torch.tensor(type_codes, dtype=torch.int16, device=intersect_lines.device)
        type_tensor = intersect_lines[:, type_col]
        #valid_mask = ~torch.isnan(type_tensor)
        mask = torch.isin(type_tensor, type_codes_tensor)
        '''valid_mask &'''
        return intersect_lines[mask]
    
    def filter_by_length(self, intersect_lines: torch.Tensor, min_len: int, max_len: int, length_col: int = 7) -> torch.Tensor:
        """按船长范围筛选线段"""
        length_tensor = intersect_lines[:, length_col]
        #valid_mask = ~torch.isnan(length_tensor)
        mask =(length_tensor >= min_len) & (length_tensor <= max_len)
        # valid_mask & 
        return intersect_lines[mask]
    
    def analyze(self):
        """主程序：统计各截面的交通流"""
        print("="*60)
        print("2025年度交通流统计分析系统")
        print("="*60)
        
        # 1. 加载数据
        print("\n[1] 正在加载AIS数据...")
        ais_data = self.load_data()
        
        # 2. 转换为张量
        print("[2] 正在转换为张量...")
        ais_tensor = self.tensor_builder(ais_data)
        print(f"    张量形状: {ais_tensor.shape}")
        
        # 3. 构建轨迹（排序去重）
        print("[3] 正在构建轨迹...")
        sorted_tensor = self.track_builder(ais_tensor)
        print(f"    去重后记录数: {sorted_tensor.shape[0]}")
        
        # 4. 提取线段
        print("[4] 正在提取航迹线段...")
        segments = self.get_route_segments(sorted_tensor)
        print(f"    线段总数: {segments.shape[0]}")
        
        # 5. 加载截面信息
        sections = self.section_list_loader(self.config)
        
        # 6. 获取船型和船长分类信息
        ship_type_mapping = self.get_ship_type_mapping(self.config)
        length_categories = self.get_length_categories(self.config)
        
        # 7. 遍历每个截面进行统计
        print("\n" + "="*60)
        print("截面交通流统计结果")
        print("="*60)
        
        results = []
        all_stats_list = []  # 用于汇总绘图
        #section为配置文件中截面信息
        for section in sections:
            # 检测相交
            intersected_nums, intersect_lines = self.line_intersect_calculate(
                segments,
                section['point1_lon'], section['point1_lat'],
                section['point2_lon'], section['point2_lat']
            )
            
            print(f"\n截面 {section['id']}: {section['name']}")
            print(f"  总穿越船舶数: {intersected_nums}")
            
            # 按船型分类统计
            print("  按船型分类:")
            ship_type_stats = {}
            type_tensor = intersect_lines[:, 6]
            sentinel_val = float(self.nan_sentinel)
            # known_type_mask 已弃用：使用哨兵值分类（-1），无需单独保存 known_type_mask
            # known_type_mask = (type_tensor != sentinel_val)

            for ship_name, type_codes in ship_type_mapping.items():
                # 跳过包含哨兵值的类别（例如 config 中的 '未知类型'），避免与后续单独统计的 '未知' 重复计数
                if any(int(c) == self.nan_sentinel for c in type_codes):
                    # 记录但不在这里计数，'未知' 将在下方统一统计
                    print(f"    跳过类别（含哨兵）: {ship_name}")
                    continue

                filtered = self.filter_by_type_codes(intersect_lines, type_codes)
                count = filtered.shape[0]
                ship_type_stats[ship_name] = count

                print(f"    {ship_name}: {count}")
                all_stats_list.append({
                    'section_id': section['id'],
                    'section_name': section['name'],
                    'category': 'ship_type',
                    'type': ship_name,
                    'count': count
                })

            unknown_type_count = (type_tensor == sentinel_val).sum().item()
            ship_type_stats['未知'] = unknown_type_count

            print(f"    未知船型 (哨兵={self.nan_sentinel}): {unknown_type_count}")
            all_stats_list.append({
                'section_id': section['id'],
                'section_name': section['name'],
                'category': 'ship_type',
                'type': '未知',
                'count': unknown_type_count
            })
            
            # 按船长分类统计
            print("  按船长分类:")
            length_stats = {}
            length_tensor = intersect_lines[:, 7]
            # known_length_mask 已弃用：使用哨兵值分类（-1），无需单独保存 known_length_mask
            # known_length_mask = (length_tensor != sentinel_val)

            for cat in length_categories:
                # 跳过专门用于标识哨兵的分类（例如 min == max == nan_sentinel），统一由 below 的 '未知' 处理
                if cat.get('min') == cat.get('max') == self.nan_sentinel:
                    print(f"    跳过长度分类（哨兵）: {cat.get('name')}")
                    continue

                filtered = self.filter_by_length(intersect_lines, cat['min'], cat['max'])
                count = filtered.shape[0]
                length_stats[cat['name']] = count

                print(f"    {cat['name']}: {count}")
                all_stats_list.append({
                    'section_id': section['id'],
                    'section_name': section['name'],
                    'category': 'length',
                    'type': cat['name'],
                    'count': count
                })

            unknown_length_count = (length_tensor == sentinel_val).sum().item()
            length_stats['未知'] = unknown_length_count

            print(f"    未知船长 (哨兵={self.nan_sentinel}): {unknown_length_count}")
            all_stats_list.append({
                'section_id': section['id'],
                'section_name': section['name'],
                'category': 'length',
                'type': '未知',
                'count': unknown_length_count
            })

            # 一致性检查：确认按船型/船长分类的合计是否等于总穿越数
            total_by_type = sum(ship_type_stats.values())
            if total_by_type != intersected_nums:
                print(f"  ⚠️ 船型统计口径不一致: 合计={total_by_type} != 总穿越={intersected_nums}")
            total_by_length = sum(length_stats.values())
            if total_by_length != intersected_nums:
                print(f"  ⚠️ 船长统计口径不一致: 合计={total_by_length} != 总穿越={intersected_nums}")

            result = {
                'section_id': section['id'],
                'section_name': section['name'],
                'total_crossing': intersected_nums,
                'ship_type_stats': ship_type_stats,
                'length_stats': length_stats,
                'data': intersect_lines
            }
            results.append(result)
        
        # 8. 创建输出目录
        output_dir = Path(self.config['output']['trajectories_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 9. 绘制和保存统计结果
        print("\n" + "="*60)
        print("生成统计图表和CSV文件")
        print("="*60)
        
        self._save_statistics(results, all_stats_list, output_dir)
        
        print("\n" + "="*60)
        print("统计完成！")
        print("="*60)
        
        return results
    
    def _save_statistics(self, results: List[Dict], all_stats_list: List[Dict], output_dir: Path):
        """保存统计结果为CSV和图表"""
        
        # 转换为DataFrame便于处理
        stats_df = pd.DataFrame(all_stats_list)
        
        # 保存完整统计CSV
        stats_csv_path = output_dir / "traffic_statistics.csv"
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 统计数据已保存: {stats_csv_path}")
        
        # 为每个截面生成图表
        for result in results:
            section_id = result['section_id']
            section_name = result['section_name']
            
            # 创建当前截面的统计DataFrame
            section_stats = stats_df[stats_df['section_id'] == section_id]
            
            # 分别处理船型和船长统计
            ship_type_df = section_stats[section_stats['category'] == 'ship_type']
            length_df = section_stats[section_stats['category'] == 'length']
            
            # 创建2个子图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"截面 {section_id}: {section_name}", fontsize=14, fontweight='bold')
            
            # 船型分布图
            if len(ship_type_df) > 0:
                sns.barplot(data=ship_type_df, x='type', y='count', ax=axes[0], hue='type', legend=False)
                axes[0].set_title('按船型分类', fontsize=12)
                axes[0].set_xlabel('船型')
                axes[0].set_ylabel('船舶数量')
                axes[0].tick_params(axis='x', rotation=45)
            
            # 船长分布图
            if len(length_df) > 0:
                sns.barplot(data=length_df, x='type', y='count', ax=axes[1], hue='type', legend=False)
                axes[1].set_title('按船长分类', fontsize=12)
                axes[1].set_xlabel('船长范围')
                axes[1].set_ylabel('船舶数量')
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            img_path = output_dir / f"section_{section_id}_statistics.png"
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            print(f"✓ 截面{section_id}统计图已保存: {img_path}")
            plt.close()
            
            # 保存该截面的详细CSV
            section_csv = output_dir / f"section_{section_id}_detail.csv"
            result_detail = pd.DataFrame([
                {'分类': '船型', '类别': k, '数量': v} 
                for k, v in result['ship_type_stats'].items()
            ] + [
                {'分类': '船长', '类别': k, '数量': v} 
                for k, v in result['length_stats'].items()
            ])
            result_detail.to_csv(section_csv, index=False, encoding='utf-8-sig')
            print(f"✓ 截面{section_id}详细数据已保存: {section_csv}")
        
        # 绘制汇总对比图（每个截面为 x 轴，显示该截面上各类别的组成）
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('全部截面对比统计', fontsize=14, fontweight='bold')

        # 各截面船型对比 —— x 轴为截面，柱子按船型分组
        ship_pivot = stats_df[stats_df['category'] == 'ship_type'].pivot_table(
            index='section_name', columns='type', values='count', fill_value=0
        )
        ship_pivot.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('各截面船型分布对比,对于ais记录', fontsize=12)
        axes[0].set_xlabel('截面')
        axes[0].set_ylabel('船舶数量')
        axes[0].tick_params(axis='x', rotation=30)
        # 图例表示不同的船型
        axes[0].legend(title='船型', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        # 在每个柱子上标注数值
        for p in axes[0].patches:
            if isinstance(p, Rectangle):
                h = p.get_height()
                if h is None:
                    continue
                try:
                    txt = f"{int(h):,}"
                except Exception:
                    txt = f"{h}"
                axes[0].annotate(txt,
                                 (p.get_x() + p.get_width() / 2., h),
                                 ha='center', va='bottom', fontsize=8)

        # 各截面船长对比 —— x 轴为截面，柱子按船长范围分组
        length_pivot = stats_df[stats_df['category'] == 'length'].pivot_table(
            index='section_name', columns='type', values='count', fill_value=0
        )
        length_pivot.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('各截面船长分布对比,对于ais记录', fontsize=12)
        axes[1].set_xlabel('截面')
        axes[1].set_ylabel('船舶数量')
        axes[1].tick_params(axis='x', rotation=30)
        # 图例表示不同的船长区间
        axes[1].legend(title='船长范围', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        # 在每个柱子上标注数值
        for p in axes[1].patches:
            if isinstance(p, Rectangle):
                h = p.get_height()
                if h is None:
                    continue
                try:
                    txt = f"{int(h):,}"
                except Exception:
                    txt = f"{h}"
                axes[1].annotate(txt,
                                 (p.get_x() + p.get_width() / 2., h),
                                 ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        
        # 保存汇总图表
        summary_img = output_dir / "海域截面汇总图.png"
        plt.savefig(summary_img, dpi=150, bbox_inches='tight')
        print(f"✓ 汇总对比图已保存: {summary_img}")
        plt.close()


def main():

    analyzer = TrafficFlowAnalyzer("config.yaml")
    results = analyzer.analyze()
    
    # 简单的结果汇总输出
    for result in results:
        print(f"\n{result['section_name']}: {result['total_crossing']} 艘船")


if __name__ == "__main__":
    main()
