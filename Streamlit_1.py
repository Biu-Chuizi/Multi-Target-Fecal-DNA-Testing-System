import streamlit as st
import tensorflow as tf
import numpy as np

# 加载内嵌的模型
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')  # 假设模型文件名为 'model.h5'
    return model

# 预测功能
def predict(model, input_data):
    input_data = np.array([input_data])  # 转换为模型所需格式
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("模型预测工具")
    
    # 加载模型
    model = load_model()

    st.write("请输入输入数据进行预测，每个参数用空格分隔：")
    input_data_str = st.text_input("输入数据", "1.2 3.4 5.6")  # 默认示例数据

    if input_data_str:
        try:
            # 将输入字符串转换为浮点数列表
            input_data = list(map(float, input_data_str.split()))

            # 执行预测
            prediction = predict(model, input_data)

            # 显示预测结果
            st.write(f"预测结果: {prediction}")
        except ValueError:
            st.error("请输入有效的数值数据（用空格分隔）")

if __name__ == "__main__":
    main()
