#!/usr/bin/env python3

from fastapi import FastAPI, Request, HTTPException
from fastapi import Query, Path
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
# RedirectResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, date
from model_util import modelD
from model_language_quality import ModelLanguageQuality

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount the directory containing the audio assets
app.mount("/assets/audio", StaticFiles(directory="assets/audio"), name="audio")
# Mount the directory containing the text assets
app.mount("/assets/text", StaticFiles(directory="assets/text"), name="text")

@app.get("/", response_class=FileResponse)
@app.get("/index.html", response_class=FileResponse)
async def get_home_page():
    return "static/index.html"

@app.get("/transcriptions/{t_id}/analyze", response_class=JSONResponse)
async def analyze_transcription(
    t_id: int = Path(..., title="Transcription ID", description="Transcription ID to analyze"),
    category: str = Query(..., title="Transcription Category", description="Transcription category is audio or text"),
    model: str = Query(default='CATBOOST', title="Model to run", description="Model to evaluate selected asset"),
    target: str = Query(default='vocab_avg', title="Criteria to evaluate", description="Criteria to evaluate ex. 'vocab_avg'"),
    test_data: str = Query(default=None, title="Test Data File", description="Use this csv file to run model on")
    ):
    if category not in [ 'audio', 'text']:
        raise HTTPException(status_code=400, detail=(f"Transcription category '{category}' invalid. "
                                                     "Choose 'audio' or 'text' ")
        )
    # see default values in global modelD dict. definition
    modelD['asset'] = t_id
    modelD['category'] = category
    modelD['model_choice'] = model
    modelD['target'] = target
    modelD['modeling_stage'] = 2 # predict stage
    if test_data:
        modelD['test_data'] = test_data
    modelO = ModelLanguageQuality(**modelD)
    # result = modelO.run_model()
    return modelO.run_model()

@app.get("/favicon.ico", response_class=FileResponse)
async def get_favicon():
    return "static/favicon.ico"

@app.get("/api", response_class=JSONResponse)
async def test_api():
	return {"message":"Hello World"}

@app.get("/my_test", response_class=FileResponse)
async def read_test_req():
	return "static/test_page.html"
