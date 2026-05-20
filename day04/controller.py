# app.py 연결하는 라우터 설정 , 웹(서버) 와 연결되는 라우터
from fastapi import APIRouter,Request,Response
router=APIRouter(prefix='/api/model')

# 서비스 호출
from service import service

# 매핑 만들기
# http://localhost:8080/api/model/learn
@router.post("/learn") # 매핑 주소 생성
async def 학습요청( request : Request ) : # Request 객체
    list = await request.json() # request에 해당하는 객체의 body값을 직접 json 변환
    print( request )
    return service.학습요청( list )

# http://localhost:8080/api/model/predict
@router.post("/predict")
async def 예측요청( car : dict ) : # body 정보를 딕셔너리 타입으로 받기
    print( car )
    return service.예측요청( car )


