# [1] 여러가지 특성의 분류 모델
import pandas as pd
df = pd.read_csv( './day05/Fish.csv' )
# 어종 7개 , Species
fish_target = df['Species']   # 정답지
# 특성 6개 , Weight,Length1,Length2,Length3,Height,Width 
fish_input = df[ ['Weight','Length1','Length2','Length3','Height','Width'] ]  # 문제지

# 훈련 / 테스트 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split( fish_input , fish_target , test_size=0.25 , random_state= 42 )

# 스케일링( 단위 통일하기 )
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit( train_input )
train_scaled = ss.transform( train_input )
test_scaled = ss.transform( test_input )

# [*] 확률적인 경사 해결법
# fit() 모델학습에서는 정답(target)도 같이 학습 중이다. 예측(y)값과 실제 정답간의 오차(기울기/가중치 조절하면서 오차 줄이기) 측정
# 예] 산꼭대기에서 내려가는 방법 중에 가장 최적의 경로 찾는 과정 = 경사 하강법( 수많은 경우의 수 계산하여 판단 )
# (1) 경사 하강법( 정확도 좋지만 학습속도가 느리다. ) vs (2) 확률 경사 하강법( SGD : 정확도 낮지만 학습속도가 빠르다. : 미니배치)

# [*] 로그 로스 / 손실 함수 , 손실( 예측과 정답의 전체 차이 )
# 로그 로스 함수는 0과 1의 확률값이 아닌 * 오차 값 * 을 측정

# [*] 에포크
# 학습 횟수

# [2] SGDClassifier , 분류모델
from sklearn.linear_model import SGDClassifier
# loss = 'log_loss' : 로스 함수
# random_state : SGD가 전체 데이터 학습이 아닌 일부 자료( 매니배치 ) 가지고 학습하는데 사용되는 분리 기준(난수값)
# max_iter : (반복)계산 횟수 # 미니 배치이므로 전체 데이터셋을 '10'이면 10 반복학습하여 모델 성공 향상 / 최적의 정확도에서 멈춤 ( 에포크 )
# tol_None : 최저그이 정확도를 찾아도 계속 반복학습 설정
sc = SGDClassifier( loss= 'log_loss' , random_state=42 , max_iter= 10 , tol=None )  # 모델객체생성
sc.fit( train_scaled , train_target )   # 모델 학습
print( sc.score( test_scaled , test_target ) ) # 0.875 로지스틱 회귀 비슷하게  # 실행마다 다르게 나옴.
print( sc.predict( test_scaled[ : 3 ] ) ) # ['Roach' 'Pike' 'Pike']

# [3] 점진적(부분) 학습( 중간학습 가능하다. )
sc.partial_fit( train_scaled , train_target )   # (위에서 이미 학습된 모델에) 10번 + 1번 => 11번 학습
print( sc.score( test_scaled , test_target ) )

# [4] 최적의 학습횟수(에포크) 찾기
sc = SGDClassifier( loss = 'log_loss' , random_state=42 ) # max_iter 생략시 1학습

train_score = []    # 학습용 정확도 
test_score = []     # 테스트용 정확도

# 정답지의 중복제거한 고유 정답만 추출
import numpy as np
classes = np.unique( train_target ) 
for i in range( 0 , 150 ) : # 0부터 150 
    sc.partial_fit( train_scaled , train_target , classes=classes ) # 1학습

    train_score.append( sc.score( train_scaled,train_target ) )
    test_score.append( sc.score( test_scaled , test_target ) )

# 정확도 시각화 # 과대적합 # 과소적합 # 최적의 에포크(반복횟수)는 학습용 과 테스트용 이 고르게 오르는 시점. 0~20 구간
import matplotlib.pyplot as plt
plt.plot( train_score , color = 'aqua' ) # 학습용 정확도 점수
plt.plot( test_score , color="#ff00f2")  # 테스트용 정확도 점수
plt.show()


# [5] hinge / SVM( 서포트 벡터 머신 )
# 기본값( 0.00001 ), 힌지 점수는 경계면( 애매/아슬 ) 에 있는 자료들을 찾는 기준점/크기
sc = SGDClassifier( loss='hinge' , max_iter=100 , random_state=42 , alpha=0.0001 )
sc.fit( train_scaled , train_target )

print( sc.score( train_scaled , train_target ) ) # 0.9159663865546218
print( sc.score( test_scaled , test_target ) ) # 0.925

# 로지스틱 회귀 : 확률 이용한 분류
# SGD(확률 경사하강법/미니배치) : loss = 'log_loss' vs loss = 'hinge'
# 경사하강법 : 손실(예측과 정답 오차) 0 가깝게 처리 하기 위한 반복 계산 ( * 딥러닝 * )
# loss = 'log_loss'
    # 도미 확률이 51% 일 때 기울기(가중치)와 절편으로 수없이 조정하여 확률 100% 만드는 방법( 경사 하강법 )
# loss = 'hinge'
    # 도미 확률이 50$ , 0인 지점이 애매/아슬한 (경계선) 자료만 가지고 확률 조정하는 방법( 경사 하강법 )