# [1] FIsh csv 가져오기
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')

# [2] Perch(농어) 만 추출
target_fish = df[ df['Species'].isin(['Perch']) ]
target_fish.info() # 56마리

# 농어의 길이/무게 추출
perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values
print( perch_length , perch_weight ) # 길이 , 무게
# '농어' 길이 에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter( perch_length , perch_weight )
plt.show()

# [3] 학습 모델 만들기 
# (1) 준비 : 학습용과 테스트용 분리한다. 왜? 모델평가에 사용된다.
from sklearn.model_selection import train_test_split
# train_test_split( 학습자료 , 정답자료 , test_size=분리비율 , random_state=분리기준난수 )
# random_state = 분리할 때 사용되는 난수값 , # 난수값에 따라 분리한다. 
# 고정값 넣어주면 항상 동일한 분리값 넣을 수 있다. # 0~32억 사이 아무거나 , 관례적으로 42 많이 씀
train_input , test_input , train_target , test_target = train_test_split( perch_length , perch_weight , test_size=0.3 , random_state=42 )

# (2) 자료형식(모양) 구성 , 대부분 2차원 사용한다.
import numpy as np
array = np.array( [ 1 , 2 , 3 , 4 ] ) # 1차원
print( array.shape ) # shape : 배열의 모양을 반환해줌 ( 행 , 열 ) , (4, ) : 4,0이라는뜻

array2 = np.array( [ [1,2],[3,4],[5,6] ] ) # 2차원
print( array2.shape ) # ( 행 , 열 ) , ( 3 , 2 )

print( train_input.shape ) # (39,)      : 1차원 배열 --> 사이킷런 모델들은 1차원배열 학습이 불가능하다.
print( train_input ) # 1차원으로 구성된 '농어' 길이
# [17.4 36.  25.  40.  39.  43.  22.  20.  22.  24.  27.5 43.  40.  24. 21.  27.5 40.  32.8 26.5 36.5 13.7 22.7 15.  37.  35.  28.7 23.5 39. 21.  23.  22.  44.  22.5 19.  37.  22.  25.6 42.  34.5]

# T1-01( zip활용 ), T1-02( column_stack 활용 ) , T2-01( reshape )   : 1차원 -> 2차원 만드는 것

# [4]
# .reshape( 행개수 , 열개수 ) :   행개수에는 -1 넣어서 자동으로 하겠다는 뜻 , 열개수는 1개
train_input = train_input.reshape( -1 , 1 )
print( train_input )
print( train_input.shape ) # (39, 1)   : 2차원이 됐다.
train_target = train_target.reshape( -1 , 1 )
test_input = test_input.reshape( -1 , 1 )

# [[17.4]
#  [36. ]
#  [25. ]
#    ...
#  [34.5]]


# [5] 모델 학습
from sklearn.neighbors import KNeighborsClassifier # K최근접이웃 모델 찾기
from sklearn.neighbors import KNeighborsRegressor # K최근접이웃 회귀 모델
knr = KNeighborsRegressor() # 모델 객체 생성
knr.fit( train_input , train_target ) # 모델 학습 , # ( 길이 , 무게 ) # '길이'에 따른 '무게' 학습
print( knr.score( test_input , test_target ) ) # 모델 평가 , 0.9929281790592219 # 회귀모델에서는 결정계수

print( test_input )         # 모델 예측할 값 : [ 8.4  18  27.5  ] # 길이

# [[  61.4]
#  [  78. ]
#  [ 248. ]
#   . . .
#   [ 126. ]
#  [  92. ]]
print( knr.predict( test_input ) ) # 모델( 무게 ) 예측 결과 
# [[  61.4]
#  [  78. ]
#  [ 248. ]
#  . . .
#  [  92. ]]

# [6] k최근접이웃 회귀는 이웃의 평균값으로 예측한다. 하이퍼라미터(k) 조절
# k = 이웃 개수 정하기
knr = KNeighborsRegressor() # 모델 객체 생성
# 임의의 길이 생성 , 임의의 물고기 길이 5부터 45까지 생성
x = np.arange( 5 , 45 ).reshape( -1 , 1 )
print( x ) # 5 ~ 44 까지의 임의의값

for k in [ 1 , 3 , 5 , 10 ]:                        
    knr.n_neighbors = k                             
    knr.fit( train_input , train_target )           
    print( knr.score(test_input , test_target) )    
    pred = knr.predict( x )                         
    print( pred )                                   

    # 시각화
    plt.scatter( train_input , train_target )
    plt.plot( x , pred )    # plot ( 선차트이면서  회귀(예측)선 )  # x = 길이  # pred = 몸무게(예측)
    plt.title( f'k={k}')
    plt.show()

# k는 이웃개수 뜻한다. k최근접 회귀는 이웃의 평균으로 예측한다.
# 