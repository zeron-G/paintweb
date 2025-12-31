from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import base64
import io
from PIL import Image

app = FastAPI()

@app.post("/api/generate")
async def generate_image(request: Request):
    try:
        data = await request.json()
        api_key = data.get("apiKey")
        user_prompt = data.get("prompt")
        ref_image_b64 = data.get("image") # 获取前端传来的 Base64

        if not api_key:
            return JSONResponse({"error": "请输入 API Key"}, status_code=400)

        # 1. 处理参考图
        pil_image = None
        if ref_image_b64:
            # 清理 base64 头部 (data:image/png;base64,...)
            if "," in ref_image_b64:
                ref_image_b64 = ref_image_b64.split(",")[1]
            
            image_bytes = base64.b64decode(ref_image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes))

        # 2. 初始化客户端
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        # 3. 构造 contents 列表 (官方文档核心要求：[文本, 图片])
        # 我们在这里优化一下 Prompt，使其更适合“虚拟试穿”场景
        final_prompt = user_prompt
        if pil_image:
             # 如果有图，则组合：[Prompt, Image]
             # 提示词增强：告诉 AI 基于这张图来生成
             contents_payload = [final_prompt, pil_image]
        else:
             # 如果没图，仅发送文本
             contents_payload = [final_prompt]

        # 4. 调用模型
        # 根据文档，推荐使用 'gemini-2.5-flash-image' (快速) 或 'gemini-3-pro-image-preview' (专业/高质量)
        # 对于产品图生图，建议优先尝试 Gemini 3 Pro，因为它对指令遵循度更高
        model_name = "gemini-2.5-flash-image" 
        
        # 这里的 config 用于设置图片比例等，不再设置 response_mime_type
        response = client.models.generate_content(
            model=model_name,
            contents=contents_payload,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(
                    aspect_ratio="1:1" # 可以是 "3:4", "16:9" 等
                )
            )
        )

        # 5. 处理返回结果
        # 官方文档写法：遍历 parts，寻找 inline_data
        if response.candidates:
            for part in response.candidates[0].content.parts:
                # 检查是否有图像数据
                if hasattr(part, 'inline_data') and part.inline_data:
                    # 获取原生 bytes 数据
                    raw_img_data = part.inline_data.data
                    # 转回 base64 发给前端
                    img_b64_str = base64.b64encode(raw_img_data).decode('utf-8')
                    return {"image": img_b64_str}
        
        return JSONResponse({"error": "模型生成完成，但未返回图像数据。可能被安全策略拦截。"}, status_code=500)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": f"后端错误: {str(e)}"}, status_code=500)
