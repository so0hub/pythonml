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

# [3] 학습 모델 만들기 , 준비
