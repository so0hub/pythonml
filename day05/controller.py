from fastapi import APIRouter, Request
from service import service

router = APIRouter(prefix='/api/model')

@router.post("/learn")
async def 학습요청(request: Request):
    user_list = await request.json()
    print("학습용 데이터 수신 완료")
    return service.학습요청(user_list)

@router.post("/predict")
async def 예측요청(user: dict):
    print("예측 데이터 수신 :", user)
    result = service.예측요청(user)
    return result