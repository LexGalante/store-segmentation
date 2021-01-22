# -*- coding: utf-8 -*-
from joblib import load
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import JSONResponse
import uvicorn
from pydantic import BaseModel


app = FastAPI(title="store-segmentation")


class StoreSegmentationRequestSchema(BaseModel):
    pass


class StoreSegmentationResponseSchema(BaseModel):
    pass


@app.post("/{model_name}")
def post(
    request: Request,
    schema: StoreSegmentationRequestSchema
) -> StoreSegmentationResponseSchema:
    pass

@app.get("/")
def get(
    request: Request
):
    pass


if __name__ == "__main__":
    uvicorn(app, debug=True)
