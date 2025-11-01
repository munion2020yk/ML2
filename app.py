import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import joblib
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. (필수) LSTM 모델 클래스 정의 ---
# train_model.py에 있는 모델 클래스와 "정확히 동일한 구조"여야 합니다.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

# --- 2. 모델 및 스케일러 로드 ---
# @st.cache_resource: 모델, 스케일러 등 리소스를 캐시 (앱 실행 시 1회만 로드)
@st.cache_resource
def load_model_and_scaler():
    """
    저장된 LSTM 모델('lstm_model.pth')과 스케일러('scaler.joblib')를 로드합니다.
    """
    # 모델 하이퍼파라미터 (train_model.py와 동일해야 함)
    INPUT_SIZE = 3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2
    
    # 모델 아키텍처 로드
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    
    # 저장된 파라미터(Weight) 로드
    try:
        # 'cuda' 장치가 없을 수 있으므로 map_location='cpu' 추가 (중요)
        model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("오류: 'lstm_model.pth' 파일을 찾을 수 없습니다.")
        st.error("GitHub 리포지토리에 'lstm_model.pth' 파일이 포함되어 있는지 확인하세요.")
        return None, None
        
    model.eval() # 예측 모드로 설정 (중요)
    
    # 스케일러 로드
    try:
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        st.error("오류: 'scaler.joblib' 파일을 찾을 수 없습니다.")
        st.error("GitHub 리포지토리에 'scaler.joblib' 파일이 포함되어 있는지 확인하세요.")
        return None, None
        
    return model, scaler

# --- 3. 데이터 로드 및 예측 수행 ---
# @st.cache_data: 반환값이 데이터(DataFrame 등)일 때 사용
@st.cache_data(ttl=600)
def load_data_and_predict(_model, _scaler):
    """
    예측에 필요한 '최근' 데이터만 로드하고, 모델로 예측을 수행합니다.
    (모델과 스케일러는 _model, _scaler로 받아서 사용)
    """
    # --- 3-1. KRX 종목 리스트 로드 (제거됨) ---
    # 하드코딩된 리스트 사용 (train_model.py와 동일)
    target_symbols = {'005930': '삼성전자', '000660': 'SK하이닉스'}
    sector_symbols = ['005930', '000660', '042700', '036930', '055550']
    sector_names = ['삼성전자', 'SK하이닉스', '한미반도체', '주성엔지니어링', '리노공업']
    
    # --- 3-2. 예측에 필요한 최근 데이터 수집 ---
    # look_back=10 이었으므로, 최근 30일치 정도 넉넉하게 받음
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    all_symbols_to_download = sector_symbols # 수정됨
    
    df_dict = {}
    for symbol in all_symbols_to_download: # 수정됨
        try:
            df_data = fdr.DataReader(symbol, start_date) # 수정됨
            if df_data.empty or 'Close' not in df_data.columns:
                st.error(f"{symbol} 데이터 로드 실패. 예측을 중단합니다.")
                return None, None, None, None
            df_dict[symbol] = df_data['Close'] # 수정됨
        except Exception as e:
            st.error(f"{symbol} 데이터 로드 실패: {e}")
            return None, None, None, None
        
    df_prices = pd.DataFrame(df_dict).fillna(method='ffill').dropna()
    
    # 데이터가 10일치 미만일 경우 에러 처리
    if len(df_prices) < 10:
        st.error("오류: 예측에 필요한 최근 10일치 데이터를 수집하지 못했습니다.")
        return None, None, None, None

    # --- 3-3. Feature Engineering (train_model.py와 동일) ---
    actual_used_symbols = list(df_dict.keys()) # 수정됨
    df_prices['Sector_Avg'] = df_prices[actual_used_symbols].mean(axis=1) # 수정됨
    features = ['005930', '000660', 'Sector_Avg']
    
    # 예측에 사용할 마지막 10일(look_back) 데이터 추출
    last_10_days_data = df_prices[features].tail(10).values
    
    # --- 3-4. 스케일링 및 예측 ---
    # 1. 스케일링 (중요: fit_transform이 아닌 transform)
    data_scaled = _scaler.transform(last_10_days_data)
    
    # 2. 텐서 변환 및 배치 차원 추가 [10, 3] -> [1, 10, 3]
    input_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
    
    # 3. 예측
    with torch.no_grad(): # 기울기 계산 안함
        prediction_scaled = _model(input_tensor) # (1, 2)
    
    # --- 3-5. 스케일 역변환 ---
    # 예측된 [SEC, Hynix] (스케일됨)
    pred_values_scaled = prediction_scaled.cpu().numpy()[0]
    
    # 역변환을 위해 (3,) 형태로 맞춰줘야 함 (Sector_Avg는 0으로)
    dummy_features = np.zeros((1, 3))
    dummy_features[0, :2] = pred_values_scaled # 예측값 2개 삽입
    
    # [SEC, Hynix, 0] -> [실제 SEC, 실제 Hynix, 실제 0]
    prediction_actual = _scaler.inverse_transform(dummy_features)[0]
    
    pred_sec = prediction_actual[0]
    pred_hynix = prediction_actual[1]
    
    # 플로팅용 데이터 (최근 150일)
    start_date_plot = (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d')
    df_plot_dict = {}
    for symbol in target_symbols.keys():
        df_plot_dict[symbol] = fdr.DataReader(symbol, start_date_plot)['Close']
    
    df_plot = pd.DataFrame(df_plot_dict).fillna(method='ffill').rename(columns={
        '005930': '삼성전자', '000660': 'SK하이닉스'
    })
    
    df_feature_info = pd.DataFrame({ # 수정됨
        'Symbol': sector_symbols,
        'Name': sector_names
    })
    
    return pred_sec, pred_hynix, df_plot, df_feature_info

# --- 4. Streamlit UI 구성 ---
st.set_page_config(page_title="반도체 주가 예측 (LSTM)", layout="wide")
st.title("📈 반도체 섹터 기반 주가 예측 (LSTM Pre-trained)")

# 1. 모델과 스케일러 로드
with st.spinner('사전 학습된 LSTM 모델과 스케일러를 로드하는 중입니다...'):
    model, scaler = load_model_and_scaler()

# 2. 메인 로직
if model is not None and scaler is not None:
    st.success('모델 및 스케일러 로드 완료!')
    
    # 3. 데이터 로드 및 예측 수행
    with st.spinner('최신 주가 데이터를 수집하고 예측을 수행하는 중입니다...'):
        pred_sec, pred_hynix, df_plot, df_feature_info = load_data_and_predict(model, scaler)

    if pred_sec is not None:
        # 4. 사이드바 - 예측 결과
        st.sidebar.header("🔮 내일 주가 예측")
        st.sidebar.write(f"({df_plot.index[-1].date()} 기준 데이터로 예측)")

        # 삼성전자
        st.sidebar.subheader("Samsung (005930)")
        last_sec = df_plot['삼성전자'].iloc[-1]
        delta_sec = (pred_sec - last_sec) / last_sec * 100
        st.sidebar.metric("예측 종가", f"{pred_sec:,.0f} 원", f"{delta_sec:.2f} %")

        # SK하이닉스
        st.sidebar.subheader("SK Hynix (000660)")
        last_hynix = df_plot['SK하이닉스'].iloc[-1]
        delta_hynix = (pred_hynix - last_hynix) / last_hynix * 100
        st.sidebar.metric("예측 종가", f"{pred_hynix:,.0f} 원", f"{delta_hynix:.2f} %")

        # 5. 메인 화면 - 차트
        st.header("주요 데이터 차트")
        tab1, tab2 = st.tabs(["삼성전자", "SK하이닉스"])

        def plot_chart(df, col, title):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name='종가'))
            fig.add_trace(go.Scatter(x=df.index, y=df[col].rolling(window=20).mean(), name='20일 이평선'))
            fig.update_layout(title=f"<b>{title}</b>", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab1:
            plot_chart(df_plot, '삼성전자', '삼성전자(005930) 종가')
        with tab2:
            plot_chart(df_plot, 'SK하이닉스', 'SK하이닉스(000660) 종가')
            
        st.info("이 예측은 삼성전자, SK하이닉스, 그리고 아래 5개 반도체 종목의 평균을 Feature로 사용한 LSTM 모델에 의해 수행되었습니다.")
        st.dataframe(df_feature_info.rename(columns={
            'Symbol': '종목코드', 'Name': '종목명'
        }))
    else:
        st.error("데이터 수집 또는 예측 중 오류가 발생했습니다.")
else:
    st.error("모델 로드에 실패하여 앱을 실행할 수 없습니다.")
