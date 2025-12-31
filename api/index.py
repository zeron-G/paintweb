from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types # 引入 types 以便构建多模态输入
import base64
import io
from PIL import Image

app = FastAPI()

@app.post("/api/generate")
async def generate_image(request: Request):
    try:
        data = await request.json()
        api_key = data.get("apiKey")
        prompt = data.get("prompt")
        ref_image_b64 = data.get("image") # 获取前端传来的参考图 Base64

        if not api_key:
            return JSONResponse({"error": "请输入 API Key"}, status_code=400)

        # 初始化客户端
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        # 构建输入内容 contents
        # 基础是提示词
        contents_payload = [prompt]

        # 如果有参考图，处理图片
        if ref_image_b64:
            # 1. 去掉前端传来的 data:image/png;base64, 前缀（如果有）
            if "," in ref_image_b64:
                ref_image_b64 = ref_image_b64.split(",")[1]
            
            # 2. 解码为 bytes
            image_bytes = base64.b64decode(ref_image_b64)
            
            # 3. 使用 Pillow 读取以验证图片有效性 (可选，但推荐)
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            # 4. 将图片封装为 Google SDK 支持的格式
            # 注意：这里取决于具体模型是否支持 input_image。
            # 如果是 Imagen 3，通常通过 edit 或 variation 接口，但这里我们尝试通用的 generate_content 多模态输入
            contents_payload.append(image_pil)

            # 修改提示词，强化指令
            # 这一步是为了让 AI 知道图片是用来做什么的
            contents_payload[0] = f"Based on the input product image, generate: {prompt}"

        # 调用生成
        # 注意：请确保你的 model 支持图像输入。
        # 如果是 Imagen 3，目前主要支持 Text-to-Image。
        # 如果使用 Gemini 2.0 Flash 等多模态模型，它通常是生成文本。
        # 这里假设你拥有访问支持图生图的模型权限。
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", # 建议尝试支持多模态更强的模型，或者原来的 gemini-2.5-flash-image
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="image/png" # 强制要求返回图片（如果是 Gemini 模型）
            )
        )

        # 处理返回数据
        # 不同的模型返回结构可能不同，这里保留原本的逻辑，并增加容错
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    return {"image": img_b64}
        
        # 如果没有直接的 inline_data，可能是生成的链接或其他格式，视具体模型而定
        return JSONResponse({"error": "模型未返回图像数据，请检查模型是否支持该功能"}, status_code=500)

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # 打印错误日志到 Vercel 后台
        return JSONResponse({"error": f"后端错误: {str(e)}"}, status_code=500)
