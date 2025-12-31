from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
import base64
import io

app = FastAPI()

@app.post("/api/generate")
async def generate_image(request: Request):
    try:
        data = await request.json()
        api_key = data.get("apiKey")
        prompt = data.get("prompt")

        if not api_key:
            return JSONResponse({"error": "请输入 API Key"}, status_code=400)

        # 初始化客户端 (动态使用用户提供的 Key)
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        # 使用最新的图像生成模型
        # 注意：模型名称需根据你账号权限调整，通常为 imagen-3.0-generate-001 或 gemini-2.0-flash 等
        response = client.models.generate_content(
            model="gemini-2.5-flash-image", 
            contents=prompt
        )

        # 处理返回的图像数据
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'inline_data') and part.inline_data:
                # 获取 base64 字符串
                img_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                return {"image": img_b64}
        
        return JSONResponse({"error": "未生成图像数据"}, status_code=500)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
