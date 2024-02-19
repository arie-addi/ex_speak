#!/usr/bin/env python3

from fastapi import FastAPI, Request
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

@app.get("/hellow", response_class=HTMLResponse)
async def read_hellow(request: Request):
    with open(f"data/output-new_{datetime.now().strftime('%Y%m%d')}.txt", "a") as file:
        file.write(f"written on {datetime.now().strftime('%H:%M:%S')}\n")
    today = date.today().strftime("%B %d, %Y")
    return (
        "<!DOCTYPE html><html>\n"
        f"<h1>Hello World {today}</h1>\n"
        "</html>"
        )

@app.get("/favicon.ico", response_class=FileResponse)
async def get_favicon():
    return "static/favicon.ico"

@app.get("/my_test", response_class=FileResponse)
async def read_test_req():
	return "static/test_page.html"
