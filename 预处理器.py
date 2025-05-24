import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import pickle

# 加载原始数据（假设包含所有20个特征）
smote_data = pd.read_csv(r"D:/桌面/新建文件夹/相关分析_L2正则化_人工合成.csv")

# 选择8个重要特征（假设已通过特征筛选确定）
selected_features = ['CW02', 'CP05', 'CP02', 'CR02', 'CS03', 'CS01', 'CS04', 'CR03']
X_selected = smote_data[selected_features]  # 仅保留8个特征

# 重新训练预处理器（基于8个特征）
scaler_8 = StandardScaler().fit(X_selected)          # 标准化
normalizer_8 = Normalizer(norm='l2').fit(X_selected) # L2正则化

# 保存新的预处理器（8维）
preprocessors_8 = {'scaler': scaler_8, 'normalizer': normalizer_8}
with open('preprocessors_8features.pkl', 'wb') as f:
    pickle.dump(preprocessors_8, f)