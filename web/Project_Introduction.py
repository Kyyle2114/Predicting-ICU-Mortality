import streamlit as st

st.set_page_config(page_title='Project Introduction',
                   layout='wide')

  
st.title('Predicting ICU Mortality - A Machine Learning Approach')

st.header('2023 Fall - Data Analytics Project')
st.write('')


st.subheader('머신러닝 기반 중환자 중증도 추정 프로젝트')
st.write('')

st.write('머신러닝 모델을 사용하여 중환자실에 입원한 환자가 3일 내 사망할 확률을 계산하며, 해당 확률을 중환자의 중증도로 추정합니다.')

st.write('모델 훈련을 위해 MIMIC-IV 데이터셋을 사용하였습니다. 환자 입원 후 6시간 이내의 데이터를 사용하였고, ChartEvents, InputEvents, OutputEvents, LabEvents 데이터를 주로 사용하였습니다.')

st.write('중증도 추정을 위해 ResNet, XGBoost, LightGBM, RandomForest, LogisticRegression을 사용하였으며, 해당 모델들은 환자의 3일 내 사망 여부를 예측하는 이진 분류 문제를 학습하였습니다. \
    \n 이후 predict_proba 메서드를 사용하여 3일 내 사망할 확률을 계산합니다.')

st.write('또한 모델 예측에 대한 이유를 확인할 수 있도록 SHAP을 사용하였으며, 전체 사망 예측에 가장 큰 영향을 주었던 특성(Global Feature Importance)과 각 환자 별 사망 예측에 가장 큰 영향을 주었던 특성 \
    (Local Feature Importance)을 확인합니다.')

url = 'https://github.com/Kyyle2114/Predicting-ICU-Mortality'

st.write('프로젝트에 사용된 코드는 [GitHub](%s)에서 자세히 확인하실 수 있습니다.' %url)

st.write('광운대학교 정보융합학부 김주혁, 장원재')

st.subheader('모델 예측 결과')
st.write('')

st.subheader('Global Feature Importance')
st.write('')