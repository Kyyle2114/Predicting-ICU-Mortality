import streamlit as st
import pandas as pd
from numpy import abs
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from shap import TreeExplainer, Explainer, DeepExplainer

st.set_page_config(page_title='Mortality Interpretation',
                   layout='wide')

st.title('Mortality Interpretation')

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
preds = None
shap_values_plot = None
idx = None
X_test = pd.read_csv('E:/DA_project/web/pages/test.csv', index_col=0)

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
        
        # Shap 
        explainer_xgb = TreeExplainer(xgb)
        shap_values_xgb = explainer_xgb.shap_values(X)
        
        explainer_lgbm = TreeExplainer(lgbm)
        shap_values_lgbm = explainer_lgbm.shap_values(X)[1]
        
        explainer_lr = Explainer(lr, X_test)
        shap_values_lr = explainer_lr.shap_values(X)
        
        explainer_rf = Explainer(rf, X_test)
        shap_values_rf = explainer_rf.shap_values(X)[1]
        
        explainer_resnet = DeepExplainer(resnet, torch.tensor(X_test.values, dtype=torch.float))
        shap_values_resnet = explainer_resnet.shap_values(torch.tensor(X.values, dtype=torch.float))
        
        shap_values = shap_values_xgb + shap_values_lgbm + shap_values_lr + shap_values_rf + shap_values_resnet
        
        # shap value -> pd.Series 
        s = pd.Series(shap_values.reshape(-1,), )
        
        # Maximum top 12 features (Local feature importance)
        idx = abs(s).sort_values()[-12:].index
        shap_values_plot = s[idx]
        shap_values_plot.index = X.columns[idx].values

    if not input_df.empty:
        st.dataframe(y.sort_index())

    with col2 :
        # column 2
        st.title('Model Output')
        
        if not input_df.empty:
           st.bar_chart(shap_values_plot)

