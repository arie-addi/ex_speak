#!/usr/bin/env python3

from fastapi import FastAPI, Request, HTTPException
from fastapi import Query, Path
from fastapi.responses import HTMLResponse, FileResponse
# RedirectResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, date

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

@app.get("/transcriptions/{t_id}/analyze")
async def analyze_transcription(
    t_id: int = Path(..., title="Transcription ID", description="Transcription ID to analyze"),
    category: str = Query(..., title="Transcription Category", description="Transcription category is audio or text")
    ):
    if category not in [ 'audio', 'text']:
        raise HTTPException(status_code=400, detail=(f"Transcription category '{category}' invalid. "
                                                     "Choose 'audio' or 'text' ")
        )
    return {'id':f'{t_id}', 'category':f'{category}', 'vocab_avg':4.5,'fluency_avg':4,'grammar_avg':3.2, 'cefr_avg':4.2}

@app.get("/favicon.ico", response_class=FileResponse)
async def get_favicon():
    return "static/favicon.ico"

@app.get("/my_test", response_class=FileResponse)
async def read_test_req():
	return "static/test_page.html"
