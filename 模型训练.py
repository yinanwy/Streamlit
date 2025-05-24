import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# 1. 加载已预处理的数据（直接使用无需再处理）
smote_data = pd.read_csv(r"D:/桌面/新建文件夹/相关分析_L2正则化_人工合成.csv")

# 2. 提取目标变量和8个特征（假设特征名已知）
selected_features = ['CW02', 'CP05', 'CP02', 'CR02', 'CS03', 'CS01', 'CS04', 'CR03']
X_smote = smote_data[selected_features]  # 直接选择8个特征
y_smote = smote_data.iloc[:, -1]         # 目标变量

# 4. 直接训练模型（数据已预处理）
model = XGBClassifier(
    objective='binary:logistic',
    importance_type='gain',
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_smote, y_smote)  # 输入数据已经是标准化+正则化后的形式

# 5. 保存模型（无需保存预处理器）
with open('xgb_model_8features.pkl', 'wb') as f:
    pickle.dump(model, f)