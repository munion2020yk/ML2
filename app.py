import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. 핵심 기능: 데이터 로드 및 모델 학습 ---
# Streamlit의 @st.cache_data를 사용해 이 무거운 함수를 캐싱합니다.
# 앱이 리로드될 때마다 다시 실행되지 않고, 캐시된 결과를 사용합니다.
@st.cache_data
def load_data_and_train():
    """
    1. FinanceDataReader로 반도체 섹터 종목을 식별합니다.
    2. 주요 종목(삼성전자, SK하이닉스)과 섹터 지수(Top 5 평균) 데이터를 수집합니다.
    3. Feature를 엔지니어링합니다.
    4. 다중 출력 회귀 모델을 학습시킵니다.
    5. 학습된 모델, 스케일러, 플로팅용 DataFrame, 예측용 최신 데이터를 반환합니다.
    """
    
    # --- 1-1. 반도체 섹터 종목 식별 ---
    df_krx = fdr.StockListing('KRX')
    sector_name = '반도체와반도체장비'
    
    # '반도체와반도체장비' 섹터의 종목들을 시가총액(Marcap) 기준으로 정렬
    semi_stocks = df_krx[
        (df_krx['Sector'] == sector_name) & 
        (df_krx['Marcap'] > 0) # 시가총액 0 이상
    ].sort_values(by='Marcap', ascending=False)
    
    # 예측 대상: 삼성전자(005930), SK하이닉스(000660)
    target_symbols = {'005930': '삼성전자', '000660': 'SK하이닉스'}
    
    # 섹터 지수(Feature)로 사용할 종목:
    # 시가총액 상위 10개 종목 중, 삼성전자와 하이닉스를 '제외한' 상위 5개
    feature_stocks = semi_stocks[
        ~semi_stocks['Symbol'].isin(target_symbols.keys())
    ].head(5)
    
    feature_symbols = feature_stocks['Symbol'].tolist()
    
    # --- 1-2. 데이터 수집 (2018년부터) ---
    start_date = "2018-01-01"
    all_symbols = list(target_symbols.keys()) + feature_symbols
    
    df_dict = {}
    for symbol in all_symbols:
        df_dict[symbol] = fdr.DataReader(symbol, start_date)['Close']
        
    df_prices = pd.DataFrame(df_dict).fillna(method='ffill').dropna()

    # --- 1-3. Feature Engineering ---
    # 1. '섹터 지수' 생성 (Feature 종목들의 평균)
    df_prices['Sector_Avg'] = df_prices[feature_symbols].mean(axis=1)
    
    # 2. 모델용 DataFrame 준비
    # Feature: 과거 5일간의 [삼성전자, 하이닉스, 섹터평균]
    # Target: 다음 날의 [삼성전자, 하이닉스]
    look_back = 5
    df_model = pd.DataFrame()
    
    # Targets (y)
    df_model['Target_SEC'] = df_prices['005930'].shift(-1)
    df_model['Target_Hynix'] = df_prices['000660'].shift(-1)
    
    # Features (X)
    for i in range(look_back):
        df_model[f'SEC_lag_{i+1}'] = df_prices['005930'].shift(i)
        df_model[f'Hynix_lag_{i+1}'] = df_prices['000660'].shift(i)
        df_model[f'Sector_lag_{i+1}'] = df_prices['Sector_Avg'].shift(i)
        
    df_model = df_model.dropna()
    
    # 3. 데이터 분리
    X = df_model.drop(['Target_SEC', 'Target_Hynix'], axis=1)
    y = df_model[['Target_SEC', 'Target_Hynix']]
    
    # --- 1-4. 모델 학습 ---
    # [TODO: LSTM] 이 부분을 PyTorch/LSTM 모델로 대체할 수 있습니다.
    # 데모를 위해 StandardScaler와 LinearRegression 파이프라인 사용
    # LinearRegression은 다중 출력(y_sec, y_hynix)을 자동으로 지원합니다.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    pipeline.fit(X, y)
    
    # --- 1-5. 예측용 최신 데이터 준비 ---
    # 가장 마지막 날의 Feature (오늘 예측에 사용)
    latest_features = X.iloc[-1:]
    
    # 플로팅용 원본 데이터
    df_plot = df_prices[['005930', '000660', 'Sector_Avg']].rename(columns={
        '005930': '삼성전자',
        '000660': 'SK하이닉스'
    })
    
    return pipeline, latest_features, df_plot, feature_stocks

# --- 2. Streamlit UI 구성 ---

# 0. 페이지 설정
st.set_page_config(page_title="반도체 주가 예측", layout="wide")
st.title("📈 반도체 섹터 기반 주가 예측 (SEC & Hynix)")

# 1. 모델과 데이터 로드 (캐시된 결과 사용)
with st.spinner('반도체 섹터 데이터를 수집하고 모델을 학습시키는 중입니다... (1회 실행)'):
    pipeline, latest_features, df_plot, feature_stocks = load_data_and_train()
    st.success('모델 및 데이터 로드 완료!')

# 2. 사이드바 (Sidebar) - 예측 결과 표시
st.sidebar.header("🔮 내일 주가 예측")
st.sidebar.write(f"({df_plot.index[-1].date()} 기준 데이터로 예측)")

# 예측 수행
# pipeline.predict()는 [Target_SEC, Target_Hynix] 2개의 값을 반환
prediction = pipeline.predict(latest_features)[0]
pred_sec = prediction[0]
pred_hynix = prediction[1]

# 삼성전자 예측 표시
st.sidebar.subheader("Samsung (005930)")
last_sec = df_plot['삼성전자'].iloc[-1]
delta_sec = (pred_sec - last_sec) / last_sec * 100
st.sidebar.metric(
    label="예측 종가",
    value=f"{pred_sec:,.0f} 원",
    delta=f"{delta_sec:.2f} %"
)

# SK하이닉스 예측 표시
st.sidebar.subheader("SK Hynix (000660)")
last_hynix = df_plot['SK하이닉스'].iloc[-1]
delta_hynix = (pred_hynix - last_hynix) / last_hynix * 100
st.sidebar.metric(
    label="예측 종가",
    value=f"{pred_hynix:,.0f} 원",
    delta=f"{delta_hynix:.2f} %"
)

# 3. 메인 화면 - 데이터 시각화
st.header("주요 데이터 차트")

tab1, tab2, tab3 = st.tabs(["삼성전자", "SK하이닉스", "반도체 섹터 지수"])

# 공통 차트 함수
def plot_chart(df, column_name, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[column_name], name='종가',
        line=dict(color='royalblue', width=2)
    ))
    df['MA20'] = df[column_name].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'], name='20일 이동평균',
        line=dict(color='orange', width=1, dash='dot')
    ))
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title='날짜', yaxis_title='주가 (KRW)',
        hovermode="x unified",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    plot_chart(df_plot, '삼성전자', '삼성전자(005930) 종가')

with tab2:
    plot_chart(df_plot, 'SK하이닉스', 'SK하이닉스(000660) 종가')

with tab3:
    plot_chart(df_plot, 'Sector_Avg', '커스텀 반도체 섹터 지수 (Top 5 평균)')
    st.info("모델 학습에 사용된 '반도체 섹터 지수'는 아래 5개 종목의 종가 평균입니다.")
    st.dataframe(feature_stocks[['Name', 'Marcap']].rename(columns={
        'Name': '종목명', 'Marcap': '시가총액(원)'
    }))

st.divider()
st.subheader("최근 데이터 (5일)")
st.dataframe(df_plot.tail())
