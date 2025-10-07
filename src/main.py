from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class DigitsData(BaseModel):
    feature_vector: list[float]

class DigitsResponse(BaseModel):
    prediction: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=DigitsResponse)
async def predict_digit(digits_features: DigitsData):
    try:
        if len(digits_features.feature_vector) != 64:
            raise HTTPException(status_code=400, detail="feature_vector must have length 64")
        prediction = predict_data([digits_features.feature_vector])
        return DigitsResponse(prediction=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))