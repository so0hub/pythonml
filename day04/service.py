import pandas as pd
from sklearn.linear_model import LinearRegression


# 훈련 세트 와 테스트 세트 분리 ,
from sklearn.model_selection import train_test_split


# 4. 서비스 :
class Service :
    def __init__(self):
        self.model = None # 초기값이 없다는 뜻

    # 1.
    def 학습요청( self , carList ):
        df = pd.DataFrame( carList )
        trian_data = df[['평균연비', '누적주행거리키로', '출고후경과월수', '사고감가건수', '소유자변경횟수']]
        target_data = df['매매가격만원'].values
        # 학습용 , 테스트용 , 학습용타깃 , 테스트용타깃 = train_test_split( 특성 , 타깃 , test_size=비율 , random_state=분리기준난수 )
        train_input , test_input , train_target , test_target = train_test_split( trian_data , target_data , test_size=0.2, random_state=42)
        lr = LinearRegression()
        lr.fit( train_input , train_target )
        print( lr.score( test_input , test_target ) )
        self.model = lr
        return True
    
    
    # 2.
    def 예측요청( self , car ) :
        # 1) 만약에 모델이 없으면
        if self.model is None :
            return "학습 모델이 없습니다."
        # 2) 만약에 모델이 있으면 입력받은 딕셔너리 리스트로 변경
        del car['차량번호ID']
        del car['매매가격만원']
        car = [ value for value in car.value() ]
        # 3) 입력받은 리스트로 모델 예측
        predict = self.model.predict( [car] )
        # 4) 예측된 값 반환
        return predict[0]
        
service = Service() # 서비스 객체 생성
