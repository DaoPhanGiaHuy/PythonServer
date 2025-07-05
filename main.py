import os
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
import io
import timm

# --------- TẢI MODEL TỰ ĐỘNG TỪ GOOGLE DRIVE (330MB) ---------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gcEDYcUVs11kHdwx3FWVPkrwHyHOEVLm"
MODEL_PATH = "model_epoch_190.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        # Google Drive có thể chuyển hướng, cần xử lý xác thực token nếu file lớn
        session = requests.Session()
        response = session.get(MODEL_URL, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
        if token:
            params = {'id': MODEL_URL.split("id=")[-1], 'confirm': token}
            response = session.get("https://drive.google.com/uc?export=download", params=params, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")

download_model()

# --------- CẤU HÌNH MODEL ---------
DEVICE = "cpu"
NUM_CLASSES = 7
LABELS = ["algae", "major_crack", "minor_crack", "peeling", "plain", "spalling", "stain"]
LABELS_VI = {
    "algae": "Mảng rêu/mốc",
    "major_crack": "Vết nứt lớn",
    "minor_crack": "Vết nứt nhỏ",
    "peeling": "Bong tróc sơn",
    "plain": "Tường thường (không lỗi)",
    "spalling": "Vữa/bê tông bị phồng rộp",
    "stain": "Vết ố/vết bẩn"
}

def load_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Gemini API config
GEMINI_API_KEY = "AIzaSyAqc-jCVHiFJFGTUjwPu6IBRNRzdmSwYLY"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

OTHER_WAYS_PHRASES = [
    "có cách nào khác không", "cách khác", "còn phương pháp nào không",
    "có giải pháp thay thế không", "phương án khác", "còn cách nào nữa không",
    "ngoài các cách trên", "cách xử lý nào tốt hơn không", "cách phòng ngừa không",
    "làm sao để không bị lại", "ưu nhược điểm các cách", "so sánh các phương pháp",
    "chi tiết hơn nữa", "cách sửa nhanh nhất", "cách tiết kiệm chi phí", "cách chuyên nghiệp hơn",
    "nên dùng vật liệu nào", "cách an toàn nhất", "cách phù hợp với nhà có trẻ nhỏ", "cách phù hợp mùa mưa",
    "cho người không chuyên", "nên tự làm hay thuê thợ", "chi phí mỗi phương án", "dụng cụ cần thiết",
    "cách xử lý khi diện tích lớn", "cách xử lý cho chung cư", "bảo trì định kỳ", "thời gian hoàn thành"
]

last_label = {"label": None, "label_vi": None}

@app.post("/detect/")
async def detect_wall_defect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = LABELS[pred.item()]
        label_vi = LABELS_VI[label]
    last_label["label"] = label
    last_label["label_vi"] = label_vi
    return {"label": label, "label_vi": label_vi}

@app.post("/chat/")
async def chat_with_gemini(req: ChatRequest):
    if last_label["label"]:
        normalized = req.message.strip().lower()
        for phrase in OTHER_WAYS_PHRASES:
            if phrase in normalized:
                if "phòng ngừa" in normalized or "không bị lại" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy hướng dẫn các biện pháp phòng ngừa, bảo trì định kỳ để tránh lỗi này tái phát, và lưu ý các lỗi thường gặp khi xử lý."
                    )
                elif "so sánh" in normalized or "ưu nhược điểm" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy liệt kê các phương pháp xử lý phổ biến, so sánh ưu nhược điểm, chi phí, độ bền, mức độ dễ làm và đưa ra lời khuyên chọn phương án phù hợp cho từng trường hợp (ví dụ nhà có trẻ nhỏ, nhà chung cư, diện tích lớn, vùng ẩm...)."
                    )
                elif "chi phí" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Ước tính chi phí các phương án xử lý, vật liệu/dụng cụ cần thiết, thời gian hoàn thành, và cách tối ưu chi phí nếu tự làm so với thuê thợ chuyên nghiệp."
                    )
                elif "an toàn" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy lưu ý các vấn đề an toàn khi xử lý lỗi này, đặc biệt với nhà có trẻ nhỏ hoặc người già, và cách sử dụng hóa chất/vật liệu an toàn."
                    )
                elif "dụng cụ" in normalized or "vật liệu" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy liệt kê chi tiết các dụng cụ, vật liệu cần thiết cho từng phương pháp xử lý, nơi có thể mua, và cách lựa chọn vật liệu phù hợp."
                    )
                elif "thời gian" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Ước tính thời gian thực hiện cho từng phương án, các bước nên làm liên tục hay có thể chia nhỏ, các mẹo rút ngắn thời gian mà vẫn đảm bảo chất lượng."
                    )
                elif "mùa mưa" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy hướng dẫn xử lý lỗi này vào mùa mưa, lưu ý điểm khác biệt so với mùa khô, các rủi ro có thể gặp phải và cách phòng tránh."
                    )
                elif "chung cư" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy hướng dẫn xử lý lỗi này ở chung cư, lưu ý các quy định về xây dựng/cải tạo, và các mẹo khắc phục khi không thể tự ý thi công lớn."
                    )
                elif "tự làm" in normalized or "không chuyên" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy hướng dẫn cách đơn giản, dễ hiểu nhất cho người chưa có kinh nghiệm hoặc lần đầu tự sửa chữa tại nhà."
                    )
                elif "thuê thợ" in normalized:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Khi nào nên thuê thợ chuyên nghiệp, chi phí dự kiến, các tiêu chí chọn thợ/sửa chữa uy tín và lưu ý khi giám sát thi công."
                    )
                else:
                    prompt = (
                        f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                        f"{req.message} "
                        "Hãy đưa ra các giải pháp thay thế, phân tích ưu nhược điểm từng phương pháp, "
                        "và đừng lặp lại các bước đã hướng dẫn trước đó."
                    )
                break
        else:
            prompt = (
                f"Tường nhà tôi bị lỗi: {last_label['label_vi']}. "
                f"{req.message} "
                "Hãy hướng dẫn cách sửa chi tiết, dễ hiểu cho người không chuyên."
            )
    else:
        prompt = f"Tôi chưa gửi ảnh lỗi tường, nhưng câu hỏi của tôi là: {req.message}"

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(GEMINI_URL, headers=headers, json=data)
    if r.status_code == 200:
        try:
            return {"answer": r.json()['candidates'][0]['content']['parts'][0]['text']}
        except Exception:
            return {"answer": "Không thể phân tích kết quả từ Gemini."}
    else:
        return {"answer": f"Gọi API Gemini thất bại ({r.status_code}), kiểm tra lại API Key hoặc quota."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)