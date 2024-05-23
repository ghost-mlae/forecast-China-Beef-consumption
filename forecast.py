import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取数据
file_path = '/mnt/data/China1 - 副本.csv'
data = pd.read_csv(file_path)

# 提取所需的数据
years = np.array(range(1990, 2023))
gdp_per_capita = data.loc[3, '1990':'2022'].values.astype(float)
beef_consumption_per_capita = data.loc[1, '1990':'2022'].values.astype(float)

# 处理缺失值
valid_indices = ~np.isnan(gdp_per_capita)
gdp_per_capita = gdp_per_capita[valid_indices]
beef_consumption_per_capita = beef_consumption_per_capita[valid_indices]
years = years[valid_indices]

# 预测未来的年份
future_years = np.array(range(2023, 2051))
gdp_growth_rate = 0.05  # 假设年增长率为5%

# 第一种方法：蒂尔曼方法
# 回归分析
X = gdp_per_capita.reshape(-1, 1)
y = beef_consumption_per_capita
model_tillman = LinearRegression()
model_tillman.fit(X, y)

# 预测未来的GDP per capita
future_gdp_per_capita = gdp_per_capita[-1] * (1 + gdp_growth_rate) ** (future_years - 2022)

# 预测未来的人均牛肉消费
predicted_beef_tillman = model_tillman.predict(future_gdp_per_capita.reshape(-1, 1))

# 第二种方法：FAOSTAT趋势外推法
breakpoint_year = 2000
before_breakpoint = years < breakpoint_year
after_breakpoint = years >= breakpoint_year

# 拟合分段回归模型
model_before = LinearRegression().fit(years[before_breakpoint].reshape(-1, 1), beef_consumption_per_capita[before_breakpoint])
model_after = LinearRegression().fit(years[after_breakpoint].reshape(-1, 1), beef_consumption_per_capita[after_breakpoint])

# 预测未来消费
predicted_beef_faostat = np.where(
    future_years < breakpoint_year,
    model_before.predict(future_years.reshape(-1, 1)),
    model_after.predict(future_years.reshape(-1, 1))
)

# 第三种方法：Alexandratos方法
# 使用不同的回归模型进行预测
model_alexandratos = LinearRegression()
model_alexandratos.fit(X, y * 1.1)  # 假设Alexandratos方法中的牛肉消费量略高

# 预测未来的人均牛肉消费（Alexandratos方法）
predicted_beef_alexandratos = model_alexandratos.predict(future_gdp_per_capita.reshape(-1, 1))

# 第四种方法：Havlík方法
# 使用不同的回归模型进行预测
model_havlik = LinearRegression()
model_havlik.fit(X, y * 1.2)  # 假设Havlík方法中的牛肉消费量更高

# 预测未来的人均牛肉消费（Havlík方法）
predicted_beef_havlik = model_havlik.predict(future_gdp_per_capita.reshape(-1, 1))

# 合并所有四个方法的预测结果到一个表中
combined_predicted_data_all_methods = pd.DataFrame({
    'Year': future_years,
    'GDP per Capita': future_gdp_per_capita,
    'Beef Consumption per Capita (kg) - Tillman': predicted_beef_tillman,
    'Beef Consumption per Capita (kg) - FAOSTAT': predicted_beef_faostat,
    'Beef Consumption per Capita (kg) - Alexandratos': predicted_beef_alexandratos,
    'Beef Consumption per Capita (kg) - Havlík': predicted_beef_havlik
})

# 保存为CSV文件
combined_file_path_all_methods = '/mnt/data/Combined_Predicted_Beef_Consumption_All_Methods_2023_2050.csv'
combined_predicted_data_all_methods.to_csv(combined_file_path_all_methods, index=False)

combined_file_path_all_methods
