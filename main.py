from fastapi import FastAPI, Request, Response
from routes import crime, shoes, crime_process, search
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, request: Request) -> Response:
        response = await super().get_response(path, request)
        response.headers["Cache-Control"] = "no-store"
        return response


app = FastAPI(debug=True)

app.mount(
    "/crime_images",
    NoCacheStaticFiles(directory="static/crime_images"),
    name="crime_images",
)

app.mount(
    "/crime_history",
    StaticFiles(directory="static/crime_history"),
    name="crime_history",
)

app.mount(
    "/shoes_images/B",
    StaticFiles(directory="static/shoes_images/B"),
    name="shoes_images/B",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React 앱 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(crime.router)
app.include_router(crime_process.router)
app.include_router(search.router)
app.include_router(shoes.router)
