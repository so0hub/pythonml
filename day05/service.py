import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

class Service :
    def __init__(self):
        self.model = None
        self.ss = None

    def 학습요청(self, userList):
        try:
            print("\n🚨 [긴급점검] 자바가 보낸 원본 데이터 앞부분 :", userList[:2])
            df = pd.DataFrame(userList)
            
            if 'category' not in df.columns:
                print("❌ [오류] 자바가 보낸 데이터에 'category' 필드명이 없습니다!")
                return False
                
            df = df[df['category'].notna()]
            
            if len(df) == 0:
                print("❌ [오류] 자바가 보낸 데이터의 'category' 값이 전부 null입니다!")
                return False

            fashion_input = df[['age', 'gender', 'inflow', 'style']]
            fashion_target = pd.to_numeric(df['category'], errors='coerce').fillna(0).astype(int).values
            
            train_input, test_input, train_target, test_target = train_test_split( 
                fashion_input, fashion_target, test_size=0.2, random_state=42 
            )
            
            self.ss = StandardScaler()
            self.ss.fit(train_input)
            train_scaled = self.ss.transform(train_input)
            test_scaled = self.ss.transform(test_input)

            self.model = SGDClassifier(loss='log_loss', random_state=42, max_iter=100, tol=None)
            self.model.fit(train_scaled, train_target)

            accuracy = self.model.score(test_scaled, test_target)
            print(f"🏆 학습 성공! 최종 검증 정확도 : {accuracy * 100:.2f}%\n")
            return True
            
        except Exception as e:
            print(f"❌ [학습 중 예상치 못한 에러 발생] : {e}")
            return False

    def 예측요청(self, user):
        if self.model is None or self.ss is None:
            return "학습 모델이 없습니다. 먼저 학습을 진행해 주세요."
        try:    
            user_data = [int(user['age']), int(user['gender']), int(user['inflow']), int(user['style'])]
            user_scaled = self.ss.transform([user_data])
            predict = self.model.predict(user_scaled)
            return int(predict[0])
        except Exception as e:
            return f"예측 오류 발생: {e}"

service = Service()