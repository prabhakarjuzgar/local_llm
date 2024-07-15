from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.views import v1_router

def get_application() -> FastAPI:
    app = FastAPI(
        swagger_ui_parameters={"defaultModelsEpandDepth": -1},
        title="RagLocal"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.include_router(v1_router)

    return app

app = get_application()
