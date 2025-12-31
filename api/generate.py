from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

@app.post("/api/generate")
async def generate(request: Request):
    data = await request.json()
    api_key = data.get("apiKey")
    prompt = data.get("prompt")

    # 注意：根据官方文档，Imagen 模型用于生成图像
    # 端点通常为 imagen-3.0-generate-001 或类似名称
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={api_key}"
    
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }

    try:
        response = requests.post(url, json=payload)
        res_data = response.json()
        
        if "error" in res_data:
            return JSONResponse({"error": res_data["error"]["message"]}, status_code=400)
            
        # 返回图片 Base64 数据
        return res_data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)