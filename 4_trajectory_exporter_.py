import pandas as pd
import torch
import numpy as np
from pathlib import Path
from app import TrafficFlowAnalyzer

class AprilTrajectoryExporter(TrafficFlowAnalyzer):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path)

    def export_april_crossing_data(self):
        """核心逻辑：导出2024年4月穿越截面的船舶全月轨迹记录"""
        print("="*60)
        print("开始")
        print("="*60)

        # 1. 定义4月份的时间戳范围 (Unix Timestamp)
        start_ts = pd.Timestamp('2024-04-01').timestamp()
        end_ts = pd.Timestamp('2024-05-01').timestamp() # 5月1日之前

        # 2. 加载数据并构建原始张量
        ais_data = self.load_data()
        raw_tensor = self.tensor_builder(ais_data)
        
        # 3. 筛选 2024年4月 的记录（用于寻找在该月穿越的船舶）
        # time 在第0列
        april_mask = (raw_tensor[:, 0] >= start_ts) & (raw_tensor[:, 0] < end_ts)
        april_tensor = raw_tensor[april_mask]
        print(f"4月份原始记录数: {april_tensor.shape[0]}")
        
        if april_tensor.shape[0] == 0:
            print("错误：数据集中没有2024年4月份的记录。")
            return

        # 4. 构建轨迹并获取线段
        # 这里使用 april_tensor 来找在该月穿越的船
        sorted_april_tensor = self.track_builder(april_tensor)
        segments = self.get_route_segments(sorted_april_tensor)
        
        # 5. 加载截面信息
        sections = self.section_list_loader(self.config)
        
        # 创建输出目录
        output_dir = Path(self.config['output']['trajectories_dir']) / "4_Trajectories"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. 遍历截面处理
        # 即使配置文件里有更多，我们也只取前4个
        for section in sections[:4]:
            print(f"\n正在处理截面 {section['id']}: {section['name']}")
            
            # 检测相交线段
            _, intersect_lines = self.line_intersect_calculate(
                segments,
                section['point1_lon'], section['point1_lat'],
                section['point2_lon'], section['point2_lat']
            )
            
            if intersect_lines.shape[0] == 0:
                print(f"  --> 4月份无船舶穿越此截面。")
                continue
                
            # 获取所有穿越该截面的船舶 MMSI (去重)
            # intersect_lines 列顺序：0=mmsi, 1=start_lon...
            crossing_mmsis = torch.unique(intersect_lines[:, 0])
            print(f"  --> 穿越船舶总数: {len(crossing_mmsis)}")
            
            # 7. 匹配：从 4月份所有记录中 提取这些 MMSI 的全月轨迹
            # 使用 torch.isin 在 GPU/CPU 上快速匹配
            mmsi_match_mask = torch.isin(sorted_april_tensor[:, 1], crossing_mmsis)
            final_trajectories = sorted_april_tensor[mmsi_match_mask]
            
            # 8. 转换并导出为 CSV
            columns = ['time', 'mmsi', 'lon', 'lat', 'type', 'length']
            df_export = pd.DataFrame(final_trajectories.cpu().numpy(), columns=columns)
            
            # 时间转换回可读格式（可选，为了方便可视化工具直接读取）
            df_export['time_readable'] = pd.to_datetime(df_export['time'], unit='s')
            
            output_path = output_dir / f"April_Section_{section['id']}_Trajectories.csv"
            df_export.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"  --> 已保存 {len(df_export)} 条轨迹记录至: {output_path}")

        print("\n" + "="*60)
        print("所有截面的4月轨迹数据提取完成！")
        print("="*60)

def main():
    exporter = AprilTrajectoryExporter("config.yaml")
    exporter.export_april_crossing_data()

if __name__ == "__main__":
    main()