from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference.pipeline import InferencePipeline

load_dotenv()

pipeline: Optional[InferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = InferencePipeline()
    yield
    pipeline = None


app = FastAPI(lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class InferRequest(BaseModel):
    image: str
    model: str


class LoadModelRequest(BaseModel):
    model: str


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    models = pipeline.list_models()
    return {
        "models": models,
        "current": pipeline.current_model,
    }


@app.get("/api/info")
def info() -> Dict[str, Any]:
    provider = pipeline.providers[0].replace("ExecutionProvider", "").lower()
    return {"device": provider}



def select_model(body: LoadModelRequest) -> Dict[str, str]:
    try:
        pipeline.load_model(body.model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"loaded": body.model}


@app.post("/api/infer")
def infer(body: InferRequest) -> Dict[str, Any]:
    if pipeline.current_model != body.model:
        try:
            pipeline.load_model(body.model)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    try:
        result = pipeline.infer_base64(body.image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
