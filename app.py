import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

@st.cache_resource
def load_resources():
    # 加载8维预处理器和模型
    with open('xgb_model_8features.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessors_8features.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer, preprocessors

model, explainer, preprocessors = load_resources()
scaler = preprocessors['scaler']  # 8维标准化器
normalizer = preprocessors['normalizer']  # 8维正则化器

# 用户输入界面（仅8个特征）
# 用户输入界面
st.title('企业信用风险预测')
st.markdown("请输入指标：")

# 输入字段
CW02 = st.number_input("注册资本(万)")
CP05 = st.number_input("融资历程(次数)")
CP02 = st.number_input("专利数 ")
CR02 = st.number_input("历史处置记录")
CS03 = st.number_input("纳税员工数 ")
CS01 = st.number_input("企业规模(1=微型，2=小型，3=中型，4=大型)")
CS04 = st.number_input("对外投资数")
CR03 = st.number_input("关联异动监控")


if st.button('失信概率'):
    input_data = pd.DataFrame([[CW02, CP05, CP02, CR02, CS03, CS01, CS04, CR03]],
                              columns=['CW02', 'CP05', 'CP02', 'CR02', 'CS03', 'CS01', 'CS04', 'CR03'])
    input_scaled = scaler.transform(input_data)
    input_processed = normalizer.transform(input_scaled)

    prob = model.predict_proba(input_processed)[0, 1]
    st.success(f"**失信概率：{prob:.1%}**")

    # SHAP解释
    st.subheader("主要影响因素：")
    shap_values = explainer.shap_values(input_processed)

    # 可视化
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    st.pyplot(fig)

    # 关键因素标注
    st.markdown("**关键因素分析：**")
    feature_impact = dict(zip(input_data.columns, shap_values[0]))
    for feature, impact in feature_impact.items():
        if impact > 0:
            st.markdown(f"🔴 `{feature}` 增加风险（贡献值：{impact:.2f}）")
        else:
            st.markdown(f"🔵 `{feature}` 降低风险（贡献值：{impact:.2f}）")
