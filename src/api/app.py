import io
import sys
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse

from src.pipeline.prediction_pipeline import CustomerChurnPredictor
from src.api.schemas import BatchPredictionRequest
from src.api.services.prediction_service import PredictionService

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

predictor: CustomerChurnPredictor | None = None
service: PredictionService | None = None


@app.on_event("startup")
def load_model():
    global predictor, service
    predictor = CustomerChurnPredictor()
    service = PredictionService(predictor)


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "healthy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# CSV Batch Prediction (Streaming Output)
# ============================================================
@app.post("/v1/predict/batch-csv", tags=["prediction"])
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported")

    df = pd.read_csv(io.BytesIO(await file.read()))
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    result = service.run_batch(df)

    buffer = io.StringIO()
    result["predictions_df"].to_csv(buffer, index=False)
    buffer.seek(0)

    headers = {
        "X-Request-ID": result["request_id"],
        "X-Model-Version": result["model_version"],
    }

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers=headers,
    )


# ============================================================
# JSON Batch Prediction
# ============================================================
@app.post("/v1/predict/batch-json", tags=["prediction"])
def predict_json(payload: BatchPredictionRequest):
    df = pd.DataFrame(payload.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records provided")

    result = service.run_batch(df)

    preview = (
        result["predictions_df"]
        .head(20)
        .to_dict(orient="records")
    )

    return {
        "summary": {
            "total_records": len(result["predictions_df"]),
            "churn_rate": result["churn_rate"],
            "latency_ms": result["latency_ms"],
            "model_version": result["model_version"],
            "request_id": result["request_id"],
        },
        "preview": preview,
    }