import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 加载新的数据
file_path = 'D:/联培/data/china beef/绘图.csv'
data = pd.read_csv(file_path)



# 设置正确的索引并转置
data.set_index('country', inplace=True)
data_t = data.T

# 去掉列名中的前后空格
data_t.columns = data_t.columns.str.strip()

# 确保索引是数字
data_t.index = pd.to_numeric(data_t.index, errors='coerce')

# 将所有列转换为数字，强制错误为NaNs，然后填充NaNs为0
data_t = data_t.apply(pd.to_numeric, errors='coerce').fillna(0)

# 分离出2023年之后的数据
historical_data = data_t[data_t.index <= 2022]
forecast_data = data_t[data_t.index > 2022]

# 提取2031年之前的"Human consumption per capita(kg/per)-Fao"数据
historical_consumption_fao = data_t[data_t.index <= 2031]

# 绘图
fig, ax1 = plt.subplots(figsize=(14, 6))

# 绘制堆积图（仅限历史数据），按照指定顺序
countries = ['Brazil', 'Argentina', 'USA', 'Russia', 'Bolivia', 'Uruguay', 'Australia', 'New Zealand']
colors = ['#FF4500', '#FF8C00', '#FFC0CB', '#DAA520', '#1E90FF', '#32CD32', '#8A2BE2', '#FFD700']

# 绘制堆积图，以确保巴西在最底层
ax1.stackplot(historical_data.index, historical_data[countries].T, colors=colors, labels=countries)

# 绘制总线（历史数据）
ax1.plot(historical_data.index, historical_data['sum'], color='black', label='Total China beef implied emission')

# 绘制总线（预测数据）
ax1.plot(forecast_data.index, forecast_data['sum'], color='black', linestyle='--')

# 创建第二个y轴
ax2 = ax1.twinx()

# 使用样条插值将折线转换为平滑曲线
def smooth_line(x, y):
    x_new = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)
    return x_new, y_smooth

# 绘制人均牛肉消费量线（仅预测数据）
lines = [
    ('Beef Consumption per Capita (kg) - Alexandratos', 'blue'),
    ('Beef Consumption per Capita (kg) - Tillman', 'darkblue'),
    ('Beef Consumption per Capita (kg) - FAOSTAT', 'darkgreen'),
    ('Beef Consumption per Capita (kg) - Havlik', 'purple')
]

for line, color in lines:
    x_fore_smooth, y_fore_smooth = smooth_line(forecast_data.index, forecast_data[line])
    ax2.plot(x_fore_smooth, y_fore_smooth, color=color, linestyle='--', label=line)

# 绘制人均牛肉消费量线（仅限历史数据，不含预测数据部分）
line = 'Human consumption per capita(kg/per)-Fao'
color = 'olive'
x_hist_smooth, y_hist_smooth = smooth_line(historical_consumption_fao.index, historical_consumption_fao[line])
x_fore_smooth, y_fore_smooth = smooth_line(historical_consumption_fao.index, historical_consumption_fao[line])
ax2.plot(x_hist_smooth, y_hist_smooth, color=color, label=line)
ax2.plot(x_fore_smooth, y_fore_smooth, color=color, linestyle='--')



# 自定义图表
ax1.set_xlabel('Year')
ax1.set_ylabel('Land-use-change GHG emissions (Gt)')
ax2.set_ylabel('Beef Consumption per Capita (kg)')

# 设置x轴刻度为每5年一个
ax1.set_xticks(data_t.index[::5])

# 添加图例并放到年份下面
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

plt.title('Implied Carbon Emissions and Beef Consumption per Capita')

plt.tight_layout(rect=[0, 0.1, 1, 1])  # 调整rect参数确保图例不会与其他图元素重叠
plt.show()

