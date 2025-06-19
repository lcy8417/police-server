from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter(prefix="/shoes", tags=["Shoes"])


@router.post("/register")
async def register_shoes(image: UploadFile = File(...), form_data: str = Form(...)):
    pass


@router.get("")
async def get_all_shoes(page: int = 1):
    pass


@router.get("/{id}")
async def get_shoe_detail(id: int):
    pass


@router.put("/{id}/edit")
async def edit_shoe(id: int, data: dict):
    pass


@router.get("/{id}/extract")
async def extract_shoe_pattern(id: int):
    pass
