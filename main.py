from fastapi import FastAPI
from routes import crime, shoes, crime_process, search
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount(
    "/crime_images", StaticFiles(directory="static/crime_images"), name="crime_images"
)

app.mount(
    "/crime_history",
    StaticFiles(directory="static/crime_history"),
    name="crime_history",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React 앱 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crime.router)
app.include_router(crime_process.router)
app.include_router(search.router)
app.include_router(shoes.router)
