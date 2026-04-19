import geopandas as gpd
import pandas as pd
from utils import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# 1. 加载世界地图数据
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 2. 模拟国家、项目和影响程度数据
data = pd.read_csv('./data/influence.csv')

# 3. 将国家数据与地图数据合并
world = world.merge(data, left_on='name', right_on='NOC', how='left')

# 4. 检查图片距离的函数
def is_too_close(new_pos, placed_positions, min_distance=6.0):
    """检查新位置与已放置位置的距离"""
    for pos in placed_positions:
        distance = ((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2) ** 0.5
        if distance < min_distance:
            return True
    return False

# 4. 绘制热力图
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(column='Investment',
           cmap='coolwarm',  # 热力图配色方案
           legend=True,
           legend_kwds={'label': "Influence Level", 'orientation': "horizontal"},
           ax=ax,
           missing_kwds={'color': 'lightgrey', 'label': 'No Data'})  # 无数据国家显示灰色
# 4. 添加图片到指定位置
def add_image(ax, img_path, x, y, zoom=0.15):
    img = Image.open(img_path).convert("RGBA")
    alpha = img.split()[3]  # 获取 alpha 通道
    alpha = alpha.point(lambda p: p * 0.8)  # 调整透明度
    img.putalpha(alpha)  # 应用调整后的透明度
    img.thumbnail((180, 150))
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

def image_paths(sport):
    return f"photo/{sport}.jpg"

placed_positions = []
for _, row in data.iterrows():
    try:
        min_distance = 10.0  # 自定义最小距离阈值
        Latitude, Longitude = map_coordinates(row['NOC'])
        new_position = (Latitude, Longitude)
        if not is_too_close(new_position, placed_positions, min_distance=min_distance):
            add_image(ax, image_paths(row['Sport']), Longitude, Latitude )
            placed_positions.append(new_position)
    except Exception as e:
        print(e)
        print(row['NOC'])
ax.set_title('Global Influence of Sports', fontsize=16)
plt.show()
