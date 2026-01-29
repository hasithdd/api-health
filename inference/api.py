from fastapi import FastAPI
from inference.schemas import PredictRequest, PredictResponse
from inference.service import predict
from inference.logging import setup_logger

logger = setup_logger()

app = FastAPI(
    title="Cybersecurity Threat Detection API",
    version="1.0.0",
)

# Health Check
@app.get("/health")
def health():
    logger.info("Health check called")
    return {
        "status": "ok",
        "model_loaded": True,
    }

# Prediction Endpoint
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    logger.info(f"Prediction request received | protocol={request.protocol}")

    result = predict(request.dict())

    logger.info(
        f"Prediction={result['prediction']} "
        f"confidence={result['confidence']:.3f}"
    )

    return result
