import streamlit as st
import pandas as pd
import torch
import joblib

st.set_page_config(page_title='Mortality Prediction',
                   layout='wide')

st.title('Mortality Prediction')

col1, col2 = st.columns([0.4, 0.6], gap='large')

xgb = joblib.load('E:\DA_project\web\pages\XGBoost.pkl')
resnet = torch.load('E:\DA_project\web\pages\ResNet')
resnet.eval()
lgbm = joblib.load('E:\DA_project\web\pages\LightGBM.pkl')
rf = joblib.load('E:\DA_project\web\pages\RandomForest.pkl')
lr = joblib.load('E:\DA_project\web\pages\LogisticRegression.pkl')


input_df = pd.DataFrame()
X = None
y = None
preds_plot = None

with col1 :
    # column 1 
    st.title('Model Input')
    
    uploaded_file = st.file_uploader('Choose a file \
                                    (Support for Excel files)',
                                    type=['csv'])
    if uploaded_file is not None:
        # Data input
        input_df = pd.read_csv(uploaded_file, index_col=0)
        y = input_df.mortality_in_3days
        X = input_df.drop('mortality_in_3days', axis=1)
        
        # Model prediction (Soft Voting)
        pred_xgb = xgb.predict_proba(X)[:, 1]
        pred_lgbm = lgbm.predict_proba(X)[:, 1]
        pred_rf = rf.predict_proba(X)[:, 1]
        pred_lr = lr.predict_proba(X)[:, 1]
        pred_resnet = torch.sigmoid(resnet(torch.tensor(X.values, dtype=torch.float))).detach().numpy().reshape(-1, )
        
        # pred range in [0, 1]
        pred = pred_xgb + pred_lgbm + pred_rf + pred_lr + pred_resnet
        pred /= 5
        
        y = pd.concat([y, pd.Series(pred, index=y.index, name='Model Predict')], axis=1)
        
        preds_plot = pd.Series(pred, index=y.index)

    if not input_df.empty:
        st.dataframe(y.sort_index())

    with col2 :
        # column 2 
        st.title('Model Output')
        
        if not input_df.empty:
            st.bar_chart(preds_plot)
