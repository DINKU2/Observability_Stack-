from fastapi import FastAPI
import httpx

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "from API 1"}

@app.get("/call-api2")
async def call_api2():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://api2:8000/")
        return {"API1 got response from API2": response.json()}
