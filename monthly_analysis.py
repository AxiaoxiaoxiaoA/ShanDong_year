import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import rcParams
from app import TrafficFlowAnalyzer  # 复用原有的类

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

class MonthlyTrafficAnalyzer(TrafficFlowAnalyzer):
    def plot_monthly_statistics(self):
        """核心逻辑：按月统计并绘制高分辨率图表"""
        print("="*60)
        print("开始执行：分截面月度船型流量统计")
        print("="*60)

        # 1. 基础数据准备 (复用原程序流程)
        ais_data = self.load_data()
        ais_tensor = self.tensor_builder(ais_data)
        sorted_tensor = self.track_builder(ais_tensor)
        segments = self.get_route_segments(sorted_tensor)
        sections = self.section_list_loader(self.config)
        ship_type_mapping = self.get_ship_type_mapping(self.config)

        # 反向映射：type_code -> ship_name
        code_to_name = {}
        for name, codes in ship_type_mapping.items():
            for code in codes:
                code_to_name[float(code)] = name

        # 2. 遍历截面收集数据
        # 预设画布：4个截面，垂直排列或2x2
        fig, axes = plt.subplots(4, 1, figsize=(24, 36), dpi=300)

        for i, section in enumerate(sections[:4]):  # 只取前4个截面
            ax = axes[i]
            
            # 计算相交
            _, intersect_lines = self.line_intersect_calculate(
                segments,
                section['point1_lon'], section['point1_lat'],
                section['point2_lon'], section['point2_lat']
            )


            # 转换为 DataFrame 进行处理

            df_inter = pd.DataFrame({
                'timestamp': intersect_lines[:, 5].cpu().numpy(),
                'type_code': intersect_lines[:, 6].cpu().numpy()
            })

            # 时间转换与月份提取
            df_inter['date'] = pd.to_datetime(df_inter['timestamp'], unit='s')
            df_inter['月份'] = df_inter['date'].dt.month
            df_inter['船型'] = df_inter['type_code'].map(code_to_name).fillna('其他/未知')

            # 统计：每个月、每种船型的数量
            monthly_stats = df_inter.groupby(['月份', '船型']).size().reset_index(name='穿越数')
            
            # 确保1-12月都在横轴上
            all_months = pd.DataFrame({'月份': range(1, 13)})
            
            # 绘图
            sns.barplot(
                data=monthly_stats, 
                x='月份', 
                y='穿越数', 
                hue='船型', 
                ax=ax,
                palette='bright',
                edgecolor='black'
            )

            # 设置标题和标签（超大字号增加可读性）
            ax.set_title(f"截面 {section['id']}: {section['name']} - 2025年度月度流量统计", fontsize=35, pad=20)
            ax.set_xlabel("月份", fontsize=25)
            ax.set_ylabel("各船型穿越数 (架次)", fontsize=25)
            ax.legend(title="船型种类", title_fontsize='20', fontsize='18', loc='upper right')
            ax.tick_params(axis='both', which='major', labelsize=20)

            # 在柱状图上添加数值标注
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(f'{int(p.get_height())}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), 
                                textcoords = 'offset points',
                                fontsize=12, rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存
        output_path = Path(self.config['output']['trajectories_dir']) / "月度分截面统计图.png"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"\n[成功] 高分辨率月度统计图已保存至: {output_path}")

def main():
    # 使用原有的配置文件启动
    analyzer = MonthlyTrafficAnalyzer("config.yaml")
    analyzer.plot_monthly_statistics()

if __name__ == "__main__":
    main()