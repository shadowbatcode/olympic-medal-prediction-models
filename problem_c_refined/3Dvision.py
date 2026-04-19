import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
compete_data = pd.read_csv('./data/compete_prob.csv')[:20]

compete_data = compete_data.sort_values(by='Expected_Medals', ascending=False).reset_index(drop=True)
midpoint = len(compete_data) // 2
compete_data['Group'] = ['Inner'] * midpoint + ['Outer'] * (len(compete_data) - midpoint)

# 提取内外组
inner_group = compete_data[compete_data['Group'] == 'Inner']
outer_group = compete_data[compete_data['Group'] == 'Outer']

# X轴编号（只显示外侧的国家标签）
x_outer = outer_group.index
x_inner = inner_group.index

# Y轴组别（Outer=1，Inner=0）
y_outer = np.ones(len(outer_group))
y_inner = np.zeros(len(inner_group))

# Z轴获奖概率
z_outer = np.zeros(len(outer_group))
z_inner = np.zeros(len(inner_group))

# 柱宽和高度
dx = dy = 0.5
dz_outer = outer_group['Expected_Medals']
dz_inner = inner_group['Expected_Medals']

# 创建三维图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制外侧柱状图
ax.bar3d(x_outer, y_outer, z_outer, dx, dy, dz_outer, color='skyblue', edgecolor='white', label='Outer Group')

# 绘制内侧柱状图
ax.bar3d(x_outer, y_inner, z_inner, dx, dy, dz_inner, color='orange', edgecolor='white', label='Inner Group')

# 添加内侧柱顶端的国家标签
for i, (x, z, country) in enumerate(zip(x_outer, dz_outer, inner_group['NOC'])):
    ax.text(x+0.2, y_outer[0]+0.2, z + x*0.01+0.5, country, color='black', ha='center', fontsize=9)

# 设置X轴，仅显示外侧的国家标签
ax.set_xticks(x_outer)
ax.set_xticklabels(outer_group['NOC'], rotation=45, ha='right')

# 设置Y轴
ax.set_yticks([0, 1])
ax.set_ylabel('Group')

# 设置Z轴
ax.set_zlabel('Medal Probability')

# 自定义句柄（解决 legend 问题）
handles = [plt.Rectangle((0, 0), 1, 1, color='skyblue', edgecolor='white', label='Inner Group'),
           plt.Rectangle((0, 0), 1, 1, color='orange', edgecolor='white', label='Outer Group')]
ax.legend(handles=handles, loc='upper right')

# 显示图形
plt.tight_layout()
plt.title('Possibility of Winning a Medal by Country in the Olympics')
plt.show()