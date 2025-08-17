import numpy as np
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_resources():
    # 加载4维预处理器和模型
    with open('xgb_model_4features.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessors_4features.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer, preprocessors


model, explainer, preprocessors = load_resources()
scaler = preprocessors['scaler']  # 4维标准化器
normalizer = preprocessors['normalizer']  # 4维正则化器

# 用户输入界面
st.title('民办养老机构信用风险评估')
st.markdown("请输入指标：")

# 输入字段
CW02 = st.number_input("注册资本(万)")
CP05 = st.number_input("融资历程(次数)")
CP02 = st.number_input("专利数量 ")
CS03 = st.number_input("纳税员工数 ")

if st.button('守信等级'):
    input_data = pd.DataFrame([[CW02, CP05, CP02,CS03]],
                              columns=['CW02', 'CP05', 'CP02', 'CS03'])
    input_scaled = scaler.transform(input_data)
    input_processed = normalizer.transform(input_scaled)

    prob = model.predict_proba(input_processed)[0, 1]
    # 数值显示
    # st.success(f"**守信等级：{prob:.4%}**")

    # 根据概率值划分层次
    if prob < 0.94:
        level = "低"
        color = "red"
    elif prob < 0.995:
        level = "中"
        color = "orange"
    else:
        level = "高"
        color = "green"

    # 使用HTML标记和颜色显示结果
    # 只显示文字结果
    st.markdown(f"<p style='font-size:20px;'>守信概率：<span style='color:{color};font-weight:bold;'>{level}</span></p>",
                unsafe_allow_html=True)

    with st.expander("点击查看数据处理细节"):
        st.write("原始输入值：", input_data.values)
        st.write("标准化后：", input_scaled)
        st.write("归一化后：", input_processed)

    # SHAP解释
    st.subheader("特征影响分析：")
    # 获取基准值和SHAP值
    shap_values = explainer.shap_values(input_processed)

    # 兼容处理二分类/多分类
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[1]  # 多分类取第二类
    else:
        base_value = explainer.expected_value  # 二分类直接使用
    # 在SHAP可视化前设置字体
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    # 可视化
    plt.figure()
    shap.force_plot(
        base_value,
        shap_values[0],  # 取第一个样本的SHAP值
        input_data.iloc[0],  # 取第一个样本的特征值
        matplotlib=True
    )
    st.pyplot(plt.gcf())

    # 创建瀑布图样式的个体特征分析
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按影响大小排序
    feature_order = sorted(zip(input_data.columns, shap_values[0]),
                           key=lambda x: abs(x[1]), reverse=False)
    ordered_features = [x[0] for x in feature_order]
    ordered_values = [x[1] for x in feature_order]

    # 绘制每个特征的影响
    y_pos = range(len(ordered_features))
    colors = ['red' if val > 0 else 'green' for val in ordered_values]
    ax.barh(y_pos, ordered_values, color=colors, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_features)
    ax.set_xlabel('SHAP值 (对预测概率的影响)')
    ax.set_title('各特征对失信概率的影响')

    # 添加基准线和预测线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=prob - base_value, color='blue', linestyle=':', linewidth=1,
               label=f'预测值 ({prob:.1%})')

    # # 添加数值标签
    # for i, v in enumerate(ordered_values):
    #     ax.text(v, i, f"{v:.3f}", color='black', ha='left' if v < 0 else 'right',va='center', fontsize=9)
    # ax.legend()
    # st.pyplot(fig)

    # 显示原始输入值作为参考
    st.markdown("**当前输入值:**")
    st.dataframe(input_data.style.format("{:.1f}"))
