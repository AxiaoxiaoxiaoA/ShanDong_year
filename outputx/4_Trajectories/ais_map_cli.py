#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIS 船舶轨迹可视化工具
用法：python ais_map_cli.py <csv文件路径> [输出html路径]
"""

import pandas as pd
import numpy as np
import folium
import re
import sys
import os
import argparse
from folium.features import DivIcon
import colorsys
import yaml
from pathlib import Path


# =========================
# 预定义高对比度颜色
# =========================
PRESET_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
    "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff",
    "#00ffff", "#ff8000", "#8000ff", "#0080ff", "#ff0080"
]

# =========================
# 截面线配置
# =========================
SECTIONS = [
    {
        "id": 1,
        "name": "2025年度场址北侧及老铁山水道至成山角交通流",
        "p1": ("37°53′06\"N", "121°59′42\"E"),
        "p2": ("37°59′06\"N", "122°04′54\"E"),
        "remark": "截面交通流 / 轨迹"
    },
    {
        "id": 2,
        "name": "2025年度场址北侧2海里交通流",
        "p1": ("37°52′10\"N", "122°1′11\"E"),
        "p2": ("37°53′45\"N", "122°2′45\"E"),
        "remark": "截面交通流 / 轨迹"
    },
    {
        "id": 3,
        "name": "2025年度场址南侧及成山角至长山水道至交通流",
        "p1":  ("37°51′54\"N", "121°59′06\"E"),
        "p2": ("37°42′36\"N", "121°56′24\"E"),
        "remark": "截面交通流 / 轨迹"
    },
    {
        "id": 4,
        "name": "2025年度场址南侧2海里交通流",
        "p1":  ("37°49′54\"N", "121°59′24\"E"),
        "p2": ("37°52′48\"N", "122°01′06\"E"),
        "remark": "截面交通流 / 轨迹"
    }
]


# =========================
# 工具函数
# =========================
def dms_to_dd(dms_str:  str) -> float:
    """度分秒转十进制度"""
    nums = re.findall(r'(\d+)', dms_str)
    if len(nums) < 3:
        raise ValueError(f"无法解析 DMS: {dms_str}")
    d, m, s = map(float, nums[:3])
    sign = -1 if dms_str.strip().upper().endswith(('W', 'S')) else 1
    return sign * (d + m / 60.0 + s / 3600.0)


def generate_distinct_colors(n:  int) -> list:
    """生成 n 个高辨识度的颜色（HSV 色环均匀分布）"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.75 + (i % 3) * 0.08
        value = 0.85 + (i % 2) * 0.1
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        colors.append(hex_color)
    return colors


# =========================
# 数据处理
# =========================
def load_ais_csv(file_path: str) -> pd.DataFrame:
    """加载并预处理 AIS CSV 数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    df = pd. read_csv(file_path)

    df = df.rename(columns={
        "time":  "timestamp",
        "sog": "speed",
        "cog": "course",
        "trueHeading": "heading"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', errors="coerce")
    df = df.dropna(subset=["mmsi", "lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    
    # 去重
    df = df.drop_duplicates(subset=["mmsi", "timestamp", "lat", "lon"])

    # 确保存在 'type' 列并为数值，缺失按 NaN 保留（后续映射为未知）
    if 'type' in df.columns:
        df['type'] = pd.to_numeric(df['type'], errors='coerce')
    else:
        df['type'] = np.nan

    return df


def classify_vessel_direction(df: pd.DataFrame) -> pd.DataFrame:
    """根据轨迹首尾判定船舶上下行"""
    df = df. copy()
    direction_map = {}

    for mmsi, g in df.groupby("mmsi"):
        g = g.sort_values("timestamp")

        if len(g) < 2:
            direction_map[mmsi] = "unknown"
            continue

        lat_start = g. iloc[0]["lat"]
        lat_end = g.iloc[-1]["lat"]

        if lat_end - lat_start < -0.001:
            direction_map[mmsi] = "upstream"
        elif lat_end - lat_start > 0.001:
            direction_map[mmsi] = "downstream"
        else:
            direction_map[mmsi] = "unknown"

    df["direction"] = df["mmsi"].map(direction_map)
    return df


# =========================
# 地图绑定
# =========================
def add_vessel_legend(m, vessel_colors: dict, max_show: int = 20, title: str = '图例'):
    """在地图上添加图例（键可以是 MMSI 或 船型标签）"""
    legend_html = f'''
    <div style="position:  fixed; 
                bottom: 50px; right: 10px; 
                background-color:  white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 10px;
                z-index: 9999;
                font-size: 12px;
                max-height:  400px;
                overflow-y: auto;">
    <b>{title}</b><br>
    '''

    items = list(vessel_colors.items())
    for key, color in items[: max_show]:
        legend_html += f'<span style="color:{color}; font-size: 16px;">■</span> {key}<br>'

    if len(items) > max_show:
        legend_html += f'<i>... 共 {len(items)}</i><br>'

    legend_html += '</div>'

    m.get_root().html.add_child(folium.Element(legend_html))


def add_section_lines(m, sections:  list, line_color: str = '#ff00ff', weight: int = 8, opacity: float = 1):
    """添加截面线"""
    for s in sections:
        lat1 = dms_to_dd(s['p1'][0])
        lon1 = dms_to_dd(s['p1'][1])
        lat2 = dms_to_dd(s['p2'][0])
        lon2 = dms_to_dd(s['p2'][1])
        p1 = [lat1, lon1]
        p2 = [lat2, lon2]

        pl = folium.PolyLine(
            locations=[p1, p2],
            color=line_color,
            weight=weight,
            opacity=opacity
        ).add_to(m)

        popup_html = f"<b>{s['id']}. {s['name']}</b><br>{s.get('remark', '')}"
        folium.Popup(popup_html, max_width=300).add_to(pl)

        folium.CircleMarker(location=p1, radius=3, color=line_color, fill=True, fillOpacity=0.9).add_to(m)
        folium.CircleMarker(location=p2, radius=3, color=line_color, fill=True, fillOpacity=0.9).add_to(m)

        mid = [(lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0]
        folium.map. Marker(
            location=mid,
            icon=DivIcon(
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                html=f'<div style="font-size:12px; font-weight:bold; color:#000; background:#fff; padding:2px 6px; border-radius:4px; border:1px solid #666;">{s["id"]}</div>'
            )
        ).add_to(m)


def create_point_map(df:  pd.DataFrame, output_html: str, show_sections: bool = True):
    """创建船舶轨迹地图"""
    if df.empty:
        print("⚠️ 数据为空，无法生成地图")
        return False

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    print(f"📍 地图中心:  ({center_lat:.4f}, {center_lon:.4f})")
    print(f"📊 方向分布:\n{df['direction'].value_counts().to_string()}")

    # 创建地图
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    # 添加海图叠加层
    folium.TileLayer(
        tiles='https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
        attr='OpenSeaMap',
        name='OpenSeaMap 海图标记',
        overlay=True,
        control=True
    ).add_to(m)

    # 添加其他底图选项
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI Ocean',
        name='ESRI 海洋底图',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer("CartoDB dark_matter", name="CartoDB 深色").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    point_all = folium.FeatureGroup(name="航点 / All Points", show=True)
    track_all = folium.FeatureGroup(name="轨迹 / All Tracks", show=True)

    # 获取所有船只的 MMSI 列表
    mmsi_list = df["mmsi"]. unique().tolist()
    n_vessels = len(mmsi_list)

    # 按船型分配颜色（提高性能）：从配置读取 ship_types 与 nan_sentinel，并映射到 type_name
    try:
        cfg_path = Path(__file__).resolve().parents[2] / 'config.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
        ship_types_cfg = cfg.get('ship_types', {})
        nan_sentinel = int(cfg.get('processing', {}).get('nan_sentinel', -1))
    except Exception:
        ship_types_cfg = {}
        nan_sentinel = -1

    # 构建 code -> type_name 映射
    code_to_name = {}
    for name, codes in ship_types_cfg.items():
        for code in codes:
            try:
                code_to_name[int(code)] = name
            except Exception:
                continue
    code_to_name[int(nan_sentinel)] = '未知'

    # 映射每条记录的 type_name
    df['type_name'] = df['type'].apply(lambda x: code_to_name.get(int(x), '未知') if pd.notna(x) else '未知')

    # 每艘船的主要船型（mode）
    mmsi_type = df.groupby('mmsi')['type_name'].agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]).to_dict()

    # 为每种类型分配颜色
    types = sorted(df['type_name'].unique())
    n_types = len(types)
    if n_types <= len(PRESET_COLORS):
        type_colors = {t: PRESET_COLORS[i] for i, t in enumerate(types)}
    else:
        generated = generate_distinct_colors(n_types)
        type_colors = {t: generated[i] for i, t in enumerate(types)}

    print(f"🚢 共 {n_vessels} 艘船，按船型分配颜色，共 {n_types} 类: {types}")

    # 创建 mmsi -> color 映射（用于绘图便利）
    vessel_colors = {mmsi: type_colors.get(mmsi_type.get(mmsi, '未知'), '#888888') for mmsi in mmsi_list}
    # 轨迹绘制
    for mmsi, g in df.groupby("mmsi"):
        g = g.sort_values("timestamp")
        color = vessel_colors.get(mmsi, "#888888")

        if len(g) >= 2:
            coords = g[["lat", "lon"]]. values. tolist()

            line = folium.PolyLine(
                coords,
                color=color,
                weight=2,
                opacity=0.8,
                tooltip=f"MMSI {mmsi}"
            )
            line.add_to(track_all)

    # 航点绘制
    for _, r in df.iterrows():
        mmsi = r['mmsi']
        color = vessel_colors.get(mmsi, "#888888")

        popup_html = f"""
        <b>MMSI: </b> {mmsi}<br>
        <b>时间:</b> {r['timestamp']}<br>
        <b>纬度:</b> {r['lat']:.6f}<br>
        <b>经度:</b> {r['lon']:.6f}<br>
        <b>方向:</b> {r['direction']}<br>
        <b>颜色:</b> <span style="color:{color}">■</span> {color}
        """

        circle = folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300)
        )
        circle.add_to(point_all)

    track_all.add_to(m)
    point_all.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    
    # 添加截面线
    if show_sections:
        add_section_lines(m, SECTIONS, line_color='#FF00FF', weight=4, opacity=0.9)

    # 添加图例
    add_vessel_legend(m, vessel_colors)

    m.save(output_html)
    print(f"✅ 地图已保存:  {output_html}")
    return True


# =========================
# 命令行入口
# =========================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AIS 船舶轨迹可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例: 
  python ais_map_cli. py data.csv
  python ais_map_cli.py data.csv -o output.html
  python ais_map_cli.py data.csv --no-sections
  python ais_map_cli.py data.csv -o map.html --no-sections
        '''
    )
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='输入的 AIS CSV 文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出的 HTML 文件路径（默认为 CSV 文件名 + .html）'
    )
    
    parser.add_argument(
        '--no-sections',
        action='store_true',
        help='不显示截面线'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    csv_path = args.csv_path
    
    # 自动生成输出文件名
    if args.output:
        output_html = args.output
    else:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_html = f"{base_name}_map.html"
    
    print(f"=" * 50)
    print(f"🗺️  AIS 船舶轨迹可视化工具")
    print(f"=" * 50)
    print(f"📂 输入文件: {csv_path}")
    print(f"📄 输出文件: {output_html}")
    print(f"-" * 50)
    
    try:
        # 加载数据
        df = load_ais_csv(csv_path)
        print(f"📋 加载后记录数: {len(df)}")
        print(f"🚢 船舶数: {df['mmsi'].nunique()}")
        
        if args.verbose:
            print(f"\n📊 数据预览:")
            print(df.head())
            print(f"\n坐标范围: lat({df['lat'].min():.4f}, {df['lat'].max():.4f}), lon({df['lon'].min():.4f}, {df['lon'].max():.4f})")
        
        # 分类方向
        df = classify_vessel_direction(df)
        
        # 生成地图
        show_sections = not args.no_sections
        success = create_point_map(df, output_html, show_sections=show_sections)
        
        if success:
            print(f"=" * 50)
            print(f"🎉 完成！请在浏览器中打开:  {output_html}")
            return 0
        else:
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())