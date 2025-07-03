from fastapi import FastAPI, Request, Response
from routes import crime, shoes, crime_process, search
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os

load_dotenv()

CRIME_IMG_DIR = os.getenv("CRIME_IMG_DIR")
SHOES_IMG_DIR = os.getenv("SHOES_IMG_DIR")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, request: Request) -> Response:
        response = await super().get_response(path, request)
        response.headers["Cache-Control"] = "no-store"
        return response


app = FastAPI(debug=True)

app.mount(
    "/" + CRIME_IMG_DIR,
    NoCacheStaticFiles(directory=f"static/{CRIME_IMG_DIR}"),
    name=CRIME_IMG_DIR,
)

app.mount(
    "/crime_history",
    StaticFiles(directory="static/crime_history"),
    name="crime_history",
)

app.mount(
    "/" + SHOES_IMG_DIR,
    StaticFiles(directory=f"static/{SHOES_IMG_DIR}"),
    name=SHOES_IMG_DIR,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ['http://localhost:3000'] 등 명확히
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crime.router)
app.include_router(crime_process.router)
app.include_router(search.router)
app.include_router(shoes.router)
