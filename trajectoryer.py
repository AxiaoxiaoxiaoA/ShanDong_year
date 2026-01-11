# simple_trajectory_exporter.py - 简化轨迹导出器

import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from app import TrafficFlowAnalyzer

class SimpleTrajectoryExporter(TrafficFlowAnalyzer):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path)
        
    def export_trajectories(self, time_window_hours=1.0):
        """导出第1、第3截面轨迹"""
        print("导出轨迹数据...")
        
        # 加载数据
        ais_data = self.load_data()
        ais_tensor = self.tensor_builder(ais_data)
        sorted_tensor = self.track_builder(ais_tensor)
        segments = self.get_route_segments(sorted_tensor)
        sections = self.section_list_loader(self.config)
        
        # 创建输出目录
        output_dir = Path(self.config['output']['trajectories_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理第1和第3个截面
        for section_idx in [1, 3]: 
            section = sections[section_idx - 1]
            print(f"处理截面 {section['id']}")
            
            # 获取穿越该截面的线段
            _, intersect_lines = self.line_intersect_calculate(
                segments,
                section['point1_lon'], section['point1_lat'],
                section['point2_lon'], section['point2_lat']
            )
            
            if intersect_lines. shape[0] == 0:
                print(f"截面{section['id']}无穿越记录")
                continue
            
            # 获取所有穿越的船舶MMSI（去重）
            crossing_mmsis = torch.unique(intersect_lines[: , 0])
            print(f"穿越船舶数:  {len(crossing_mmsis)}")
            
            # 获取穿越时间范围
            crossing_times = intersect_lines[:, 5]
            min_time = torch.min(crossing_times)
            max_time = torch.max(crossing_times)
            
            # 扩展时间窗口
            time_window_seconds = time_window_hours * 3600
            time_start = min_time - time_window_seconds
            time_end = max_time + time_window_seconds
            
            print(f"时间窗口: {time_start. item():.0f} 到 {time_end. item():.0f}")
            
            # 筛选这些船舶在时间窗口内的所有记录
            mmsi_mask = torch.isin(sorted_tensor[: , 1], crossing_mmsis)  # mmsi在第1列
            time_mask = (sorted_tensor[:, 0] >= time_start) & (sorted_tensor[:, 0] <= time_end)  # time在第0列
            
            # 同时满足mmsi和时间条件的记录
            combined_mask = mmsi_mask & time_mask
            filtered_records = sorted_tensor[combined_mask]
            
            print(f"筛选后记录数: {filtered_records.shape[0]}")
            
            if filtered_records.shape[0] > 0:
                # 转换为DataFrame并保存
                columns = ['time', 'mmsi', 'lon', 'lat', 'type', 'length']
                df = pd.DataFrame(filtered_records. cpu().numpy(), columns=columns)
                
                output_path = output_dir / f"section_{section['id']}_trajectories.csv"
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"已保存 {len(df)} 条记录到 {output_path}")
                
                # 输出一些统计信息
                unique_ships = df['mmsi'].nunique()
                print(f"包含 {unique_ships} 艘船的轨迹")
            else:
                print(f"截面{section['id']}无有效记录")

def main():
    exporter = SimpleTrajectoryExporter("config.yaml")
    exporter.export_trajectories(time_window_hours=1.0)

if __name__ == "__main__":
    main()