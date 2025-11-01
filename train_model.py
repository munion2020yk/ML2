import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import joblib # 스케일러 저장을 위해
from datetime import datetime
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

print("--- 1. 데이터 로드 및 전처리 시작 ---")

# --- 1-2. 반도체 섹터 식별 및 데이터 수집 ---
# 타겟 종목 (예측 대상)
target_symbols = {'005930': '삼성전자', '000660': 'SK하이닉스'}

# Feature 종목 (섹터 평균 계산용)
# 사용자의 제안에 따라, 타겟 2개 + 안정적인 3개 종목, 총 5개로 구성
sector_symbols = ['005930', '000660', '042700', '036930', '055550']
sector_names = ['삼성전자', 'SK하이닉스', '한미반도체', '주성엔지니어링', '리노공업']

print(f"타겟 종목: {list(target_symbols.values())}")
print(f"섹터 평균 계산용 종목: {sector_names}")

start_date = "2018-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')
# all_symbols는 이제 sector_symbols와 동일
all_symbols_to_download = sector_symbols

df_dict = {}
for symbol in all_symbols_to_download:
    print(f"데이터 수집 중: {symbol}")
    df_data = fdr.DataReader(symbol, start_date, end_date)
    if df_data.empty or 'Close' not in df_data.columns:
        print(f"[오류] {symbol} 데이터 수집 실패. 이 종목을 제외하고 진행합니다.")
        continue
    df_dict[symbol] = df_data['Close']
    
df_prices = pd.DataFrame(df_dict).fillna(method='ffill').dropna()

# --- 1-3. Feature Engineering ---
# 1. '섹터 지수' 생성 (수집에 성공한 종목들로만 평균)
actual_used_symbols = list(df_dict.keys())
df_prices['Sector_Avg'] = df_prices[actual_used_symbols].mean(axis=1)

# 사용할 Feature: [삼성전자, SK하이닉스, 섹터평균]
# (이 두 종목은 타겟이므로 반드시 수집되어야 함)
if '005930' not in df_prices.columns or '000660' not in df_prices.columns:
    print("[치명적 오류] 타겟 종목(삼성전자, SK하이닉스) 데이터 수집에 실패했습니다.")
    exit()
    
features = ['005930', '000660', 'Sector_Avg']
data = df_prices[features].values

# --- 1-4. 데이터 스케일링 ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# !!! 중요: 스케일러 저장 (예측 시 동일하게 사용해야 함)
joblib.dump(scaler, 'scaler.joblib')
print("--- 'scaler.joblib' 저장 완료 ---")

# --- 1-5. 시퀀스 데이터 생성 (Look-back = 10) ---
look_back = 10
X, y = [], []
for i in range(len(data_scaled) - look_back):
    X.append(data_scaled[i:(i + look_back)])
    # Target: 다음 날의 [삼성전자, SK하이닉스] 종가
    y.append(data_scaled[i + look_back, :2]) 

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

print(f"X shape: {X.shape}") # (샘플 수, 10, 3)
print(f"y shape: {y.shape}") # (샘플 수, 2)

# --- 2. LSTM 모델 정의 (PyTorch) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # LSTM 출력(hidden_size)을 받아 Target(2개)으로 변환
        self.fc = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x):
        # h0, c0 초기화 (batch_size는 x.size(0))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 순전파
        # out: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))  
        
        # 마지막 타임 스텝의 출력만 사용
        out = self.fc(out[:, -1, :])
        return out

# 모델 하이퍼파라미터
INPUT_SIZE = 3    # 3개 Features (SEC, Hynix, Sector_Avg)
HIDDEN_SIZE = 64  # 은닉층 크기
NUM_LAYERS = 2    # LSTM 레이어 수
OUTPUT_SIZE = 2   # 2개 Targets (SEC, Hynix)

model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.MSELoss() # 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001) # 옵티마이저

print("\n--- 3. LSTM 모델 학습 시작 ---")
# --- 3. 모델 학습 ---
num_epochs = 100 # (실제로는 100~500 에포크 권장)
batch_size = 32

# 간단한 DataLoader (shuffle=True)
train_dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(num_epochs):
    for i, (sequences, targets) in enumerate(train_loader):
        # 순전파
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("--- 학습 완료 ---")

# --- 4. 모델 파라미터 저장 ---
# !!! 중요: 학습된 모델의 파라미터(weights) 저장
torch.save(model.state_dict(), 'lstm_model.pth')
print("--- 'lstm_model.pth' 저장 완료 ---")
print("\n[성공] 이제 다음 파일들을 GitHub에 업로드하세요:")
print("1. app.py (수정된 버전)")
print("2. lstm_model.pth")
print("3. scaler.joblib")
print("4. requirements.txt")