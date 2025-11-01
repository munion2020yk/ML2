import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. 모델 학습 및 데이터 준비 (캐싱) ---
# Streamlit의 @st.cache_data는 함수 결과를 캐싱하여, 앱이 리로드될 때마다
# 이 무거운 작업을 반복하지 않도록 합니다. (매우 중요!)
@st.cache_data
def load_and_train_model():
    """
    데이터를 로드하고 간단한 선형 회귀 모델을 학습시킵니다.
    실제 LSTM 모델을 사용하려면 이 함수 내부에서
    미리 학습된 model.pth와 scaler.pkl을 로드해야 합니다.
    """
    # 1. 데이터 로드 (삼성전자, 2015년부터)
    df = fdr.DataReader('005930', '2015-01-01')
    
    # 2. 간단한 Feature Engineering (Time-Step 흉내)
    # LSTM의 'look_back' (window) 개념을 흉내 냅니다.
    # 과거 10일간의 종가(Close)를 Feature로 사용합니다.
    look_back = 10
    df_model = df[['Close']].copy()
    
    # 'Target'은 오늘 종가
    df_model['Target'] = df_model['Close'].shift(-1) # 다음날 종가를 예측
    
    # Feature는 과거 10일간의 종가
    for i in range(look_back):
        df_model[f'lag_{i+1}'] = df_model['Close'].shift(i)
    
    # 결측치 제거
    df_model.dropna(inplace=True)
    
    # 3. 데이터 분리 및 스케일링
    X = df_model.drop(['Close', 'Target'], axis=1)
    y = df_model['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. 모델 학습 (LinearRegression으로 대체)
    # *************************************************************
    # * 실제 프로젝트에서는 이 부분에
    # * 미리 학습된 PyTorch LSTM 모델(model.pth)을 로드하는 코드를
    # * 작성해야 합니다.
    # * model = LSTMModel(...)
    # * model.load_state_dict(torch.load('model.pth'))
    # *************************************************************
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 5. 예측에 필요한 최신 데이터와 스케일러, 원본 DF 반환
    latest_features = X.iloc[-1:].copy() # 가장 마지막 날의 Feature
    
    return model, scaler, latest_features, df

# --- 2. Streamlit 앱 UI 구성 ---

# 0. 페이지 설정
st.set_page_config(page_title="삼성전자 주가 예측", layout="wide")
st.title("삼성전자 주가 예측 (Simple ML Demo)")

# 1. 모델과 데이터 로드 (캐시된 결과 사용)
with st.spinner('모델을 로드하고 데이터를 준비하는 중입니다...'):
    model, scaler, latest_features, df = load_and_train_model()

# 2. 사이드바 (Sidebar) - 예측 컨트롤
st.sidebar.header("📈 내일 주가 예측하기")
st.sidebar.write("과거 10일간의 데이터를 기반으로 내일의 삼성전자 종가를 예측합니다.")

if st.sidebar.button("예측 실행"):
    # 1. 최신 데이터 스케일링
    latest_features_scaled = scaler.transform(latest_features)
    
    # 2. 예측 수행
    # *************************************************************
    # * PyTorch 모델이었다면:
    # * inputs = torch.FloatTensor(latest_features_scaled).unsqueeze(0)
    # * prediction = model(inputs).item()
    # * prediction = price_scaler.inverse_transform([[prediction]])[0][0]
    # *************************************************************
    prediction = model.predict(latest_features_scaled)[0]
    
    # 3. 결과 표시
    last_close = df['Close'].iloc[-1]
    change = prediction - last_close
    change_percent = (change / last_close) * 100
    
    st.sidebar.subheader("🔮 예측 결과")
    st.sidebar.metric(
        label=f"예측 종가 ({df.index[-1].date() + timedelta(days=1)})",
        value=f"{prediction:,.0f} 원",
        delta=f"{change:,.0f} 원 ({change_percent:.2f}%)"
    )
    if change > 0:
        st.sidebar.success("상승 🔺")
    else:
        st.sidebar.error("하락 🔻")

else:
    st.sidebar.info("버튼을 눌러 예측을 시작하세요.")


# 3. 메인 화면 - 데이터 시각화
st.header("삼성전자(005930) 종가 차트")
st.write("차트를 확대/축소하거나 마우스로 드래그하여 기간을 조절할 수 있습니다.")

# Plotly로 차트 그리기
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['Close'], 
    name='종가(Close)',
    line=dict(color='royalblue', width=2)
))

# 이동평균선 추가
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA60'] = df['Close'].rolling(window=60).mean()

fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['MA20'], 
    name='20일 이동평균',
    line=dict(color='orange', width=1, dash='dot')
))
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['MA60'], 
    name='60일 이동평균',
    line=dict(color='green', width=1, dash='dot')
))

# 차트 레이아웃 업데이트
fig.update_layout(
    xaxis_title='날짜',
    yaxis_title='주가 (KRW)',
    legend_title='범례',
    hovermode="x unified",
    xaxis_rangeslider_visible=True # 하단 범위 슬라이더
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("최근 데이터")
st.dataframe(df.tail(10).sort_index(ascending=False))
