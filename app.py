from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from skimage.measure import shannon_entropy
import cv2
import numpy as np
import os
import json
import uuid

# -------------------------------------------------
# 🚀 FastAPI App
# -------------------------------------------------
app = FastAPI(title="Cow & Buffalo Breed Predictor API")

# -------------------------------------------------
# 🌐 CORS (Allow Node.js / Web Frontends)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all (safe for HF; restrict later if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 🧠 Load YOLO Model (CPU-safe for Hugging Face)
# -------------------------------------------------
MODEL_PATH = "cow_breed_yolo11_FineTune.pt"
model = YOLO(MODEL_PATH)
model.to("cpu")

# -------------------------------------------------
# 📘 Load Breed Metadata
# -------------------------------------------------
with open("breed_info.json", "r", encoding="utf-8") as f:
    breed_info = json.load(f)

# -------------------------------------------------
# 🧪 Image Quality Validator
# -------------------------------------------------
def validate_image_quality(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = np.std(gray)
    entropy = shannon_entropy(gray)

    if brightness < 40 or brightness > 210:
        return {"status": "error", "reason": "Image too dark or too bright"}
    if sharpness < 100:
        return {"status": "error", "reason": "Image too blurry"}
    if contrast < 20:
        return {"status": "error", "reason": "Low contrast"}
    if entropy < 3.5:
        return {"status": "error", "reason": "Low detail or plain background"}

    return {"status": "ok"}

# -------------------------------------------------
# 📦 Response Schema
# -------------------------------------------------
class PredictResponse(BaseModel):
    breed: str | None
    confidence: float | None
    species: str | None
    quality: dict | None

# -------------------------------------------------
# 🏠 Root Endpoint
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "✅ Cow & Buffalo Breed Predictor API is running",
        "framework": "FastAPI",
        "model": "YOLOv11 Classification",
    }

# -------------------------------------------------
# 🔮 Prediction Endpoint
# -------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):

    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        # Save uploaded image
        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        # Load image
        image = cv2.imread(temp_filename)
        if image is None:
            return {
                "breed": None,
                "confidence": None,
                "species": None,
                "quality": {"status": "error", "reason": "Invalid image file"},
            }

        # Step 1️⃣ Image Quality Check
        quality = validate_image_quality(image)
        if quality["status"] == "error":
            return {
                "breed": None,
                "confidence": None,
                "species": None,
                "quality": quality,
            }

        # Step 2️⃣ YOLO Prediction
        results = model.predict(temp_filename, verbose=False)[0]

        breed = None
        confidence = None

        if hasattr(results, "probs") and results.probs is not None:
            breed_idx = results.probs.top1
            breed = results.names[breed_idx]
            confidence = round(float(results.probs.top1conf), 4)

        elif results.boxes is not None and len(results.boxes) > 0:
            breed_idx = int(results.boxes.cls[0].item())
            breed = results.names[breed_idx]
            confidence = round(float(results.boxes.conf[0].item()), 4)

        # Step 3️⃣ Determine Species
        if breed:
            species = "Buffalo" if "buffalo" in breed.lower() else "Cow"
        else:
            species = None

        return {
            "breed": breed,
            "confidence": confidence,
            "species": species,
            "quality": quality,
        }

    except Exception as e:
        return {
            "breed": None,
            "confidence": None,
            "species": None,
            "quality": {"status": "error", "reason": str(e)},
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)





"""Previous corrected code file for reference."""

# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from ultralytics import YOLO
# from skimage.measure import shannon_entropy
# import cv2
# import numpy as np
# import os, json

# app = FastAPI()

# # ✅ CORS setup
# allowed_origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:5500",
#     "https://your-site.vercel.app",
#     "https://your-site.netlify.app"
#     "https://22b2ea38-34b1-4c67-b500-94adf0117248.lovableproject.com"  # Add this
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Load YOLO model (Render-friendly path)
# model = YOLO("cow_breed_yolo11_FineTune.pt")

# # ✅ Load breed metadata JSON
# with open("breed_info.json", "r", encoding="utf-8") as f:
#     breed_info = json.load(f)


# # ✅ Image Quality Validator
# def validate_image_quality(image: np.ndarray):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     brightness = np.mean(gray)
#     sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#     contrast = np.std(gray)
#     entropy = shannon_entropy(gray)

#     if brightness < 40 or brightness > 210:
#         return {"status": "error", "reason": "Image too dark or too bright"}
#     if sharpness < 100:
#         return {"status": "error", "reason": "Image too blurry"}
#     if contrast < 20:
#         return {"status": "error", "reason": "Low contrast"}
#     if entropy < 3.5:
#         return {"status": "error", "reason": "Low detail or plain background"}

#     return {"status": "ok"}


# # ✅ Pydantic model for response
# class PredictResponse(BaseModel):
#     breed: str | None
#     confidence: float | None
#     species: str | None
#     quality: dict | None


# # ✅ Root endpoint
# @app.get("/")
# def root():
#     return {"message": "🐄 YOLO Breed Predictor API with Image Validator is running!"}


# # ✅ Main prediction route
# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     image_path = f"temp_{file.filename}"
#     try:
#         # Save uploaded image
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Load image for validation
#         image = cv2.imread(image_path)
#         if image is None:
#             return {
#                 "breed": None,
#                 "confidence": None,
#                 "species": None,
#                 "quality": {"status": "error", "reason": "Invalid image file"},
#             }

#         # Step 1️⃣ — Image quality check
#         quality = validate_image_quality(image)
#         if quality["status"] == "error":
#             return {
#                 "breed": None,
#                 "confidence": None,
#                 "species": None,
#                 "quality": quality,
#             }

#         # Step 2️⃣ — Run YOLO classification
#         results = model.predict(image_path, verbose=False)[0]
#         breed, confidence = None, None

#         if hasattr(results, "probs") and results.probs is not None:
#             breed_idx = results.probs.top1
#             breed = results.names[breed_idx]
#             confidence = round(float(results.probs.top1conf), 4)
#         elif results.boxes is not None and len(results.boxes) > 0:
#             breed_idx = int(results.boxes.cls[0].item())
#             breed = results.names[breed_idx]
#             confidence = round(float(results.boxes.conf[0].item()), 4)

#         # ✅ Step 3️⃣ — Determine species automatically
#         if breed:
#             if "buffalo" in breed.lower():
#                 species = "Buffalo"
#             else:
#                 species = "Cow"
#         else:
#             species = None

#         return {
#             "breed": breed,
#             "confidence": confidence,
#             "species": species,
#             "quality": quality,
#         }

#     except Exception as e:
#         return {
#             "breed": None,
#             "confidence": None,
#             "species": None,
#             "quality": {"status": "error", "reason": str(e)},
#         }

#     finally:
#         # Always clean up
#         if os.path.exists(image_path):
#             os.remove(image_path)

"""Previous corrected code file for reference."""


# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from ultralytics import YOLO
# from skimage.measure import shannon_entropy
# import cv2
# import numpy as np
# import os, json

# app = FastAPI()

# # ✅ CORS setup
# allowed_origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:5500",
#     "https://your-site.vercel.app",
#     "https://your-site.netlify.app"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Load YOLO model
# model = YOLO(r"D:\\cow-breed-final\\cow_breed_yolo11_FineTune.pt")

# # ✅ Load breed metadata JSON
# with open("D:\\cow-breed-final\\breed_info.json", "r", encoding="utf-8") as f:
#     breed_info = json.load(f)


# # ✅ Image Quality Validator
# def validate_image_quality(image: np.ndarray):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     brightness = np.mean(gray)
#     sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#     contrast = np.std(gray)
#     entropy = shannon_entropy(gray)

#     if brightness < 40 or brightness > 210:
#         return {"status": "error", "reason": "Image too dark or too bright"}
#     if sharpness < 100:
#         return {"status": "error", "reason": "Image too blurry"}
#     if contrast < 20:
#         return {"status": "error", "reason": "Low contrast"}
#     if entropy < 3.5:
#         return {"status": "error", "reason": "Low detail or plain background"}

#     return {"status": "ok"}


# # ✅ Pydantic model for response
# class PredictResponse(BaseModel):
#     breed: str | None
#     confidence: float | None
#     metadata: dict | None
#     quality: dict | None


# # ✅ Root endpoint
# @app.get("/")
# def root():
#     return {"message": "🐄 YOLO Breed Predictor API with Image Validator is running!"}


# # ✅ Main prediction route
# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     image_path = f"temp_{file.filename}"
#     try:
#         # Save uploaded image
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Load image for validation
#         image = cv2.imread(image_path)
#         if image is None:
#             return {"breed": None, "confidence": None, "metadata": {}, "quality": {"status": "error", "reason": "Invalid image file"}}

#         # Step 1️⃣ — Image quality check
#         quality = validate_image_quality(image)
#         if quality["status"] == "error":
#             return {"breed": None, "confidence": None, "metadata": {}, "quality": quality}

#         # Step 2️⃣ — Run YOLO classification
#         results = model.predict(image_path, verbose=False)[0]
#         breed, confidence = None, None

#         if hasattr(results, "probs") and results.probs is not None:
#             breed_idx = results.probs.top1
#             breed = results.names[breed_idx]
#             confidence = round(float(results.probs.top1conf), 4)
#         elif results.boxes is not None and len(results.boxes) > 0:
#             breed_idx = int(results.boxes.cls[0].item())
#             breed = results.names[breed_idx]
#             confidence = round(float(results.boxes.conf[0].item()), 4)

#         # Step 3️⃣ — Get metadata
#         metadata = breed_info.get(breed, {
#             "origin": "Unknown",
#             "primary_use": "Unknown",
#             "average_milk_yield": "Unknown",
#             "average_weight": "Unknown",
#             "distribution": "Unknown",
#             "special_features": "Unknown"
#         })

#         return {
#             "breed": breed,
#             "confidence": confidence,
#             "metadata": metadata,
#             "quality": quality
#         }

#     except Exception as e:
#         return {
#             "breed": None,
#             "confidence": None,
#             "metadata": {},
#             "quality": {"status": "error", "reason": str(e)}
#         }

#     finally:
#         # Always clean up
#         if os.path.exists(image_path):
#             os.remove(image_path)


# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from ultralytics import YOLO
# import os, json

# app = FastAPI()

# # ✅ CORS setup
# allowed_origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:5500",
#     "https://your-site.vercel.app",
#     "https://your-site.netlify.app"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load YOLO model
# model = YOLO(r"cow_breed_yolo11_FineTune.pt")

# # Load breed metadata JSON
# with open("breed_info.json", "r", encoding="utf-8") as f:
#     breed_info = json.load(f)

# # Pydantic model for response
# class PredictResponse(BaseModel):
#     breed: str | None
#     confidence: float | None
#     metadata: dict | None

# # Root GET endpoint
# @app.get("/")
# def root():
#     return {"message": "YOLO Breed Predictor API is running! Use POST /predict to send images."}

# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     image_path = f"temp_{file.filename}"
#     try:
#         # Save uploaded image
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Run YOLO prediction
#         results = model.predict(image_path, verbose=False)[0]

#         # Determine top prediction
#         breed, confidence = None, None
#         if results.boxes is not None and len(results.boxes) > 0:
#             breed_idx = int(results.boxes.cls[0].item())
#             breed = results.names[breed_idx]
#             confidence = round(float(results.boxes.conf[0].item()), 4)
#         elif hasattr(results, "probs") and results.probs is not None:
#             breed_idx = results.probs.top1
#             breed = results.names[breed_idx]
#             confidence = round(float(results.probs.top1conf), 4)

#         # Get metadata
#         metadata = breed_info.get(breed, {
#             "origin": "Unknown",
#             "primary_use": "Unknown",
#             "average_milk_yield": "Unknown",
#             "average_weight": "Unknown",
#             "distribution": "Unknown",
#             "special_features": "Unknown"
#         })

#         return {
#             "breed": breed,
#             "confidence": confidence,
#             "metadata": metadata
#         }

#     except Exception as e:
#         return {
#             "breed": None,
#             "confidence": None,
#             "metadata": {},
#             "error": str(e)
#         }
#     finally:
#         # Always remove temp image
#         if os.path.exists(image_path):
#             os.remove(image_path)








# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# from ultralytics import YOLO
# import os

# app = FastAPI()

# model = YOLO(r"cow_breed_yolo11_FineTune.pt")

# # Pydantic model for response
# class PredictResponse(BaseModel):
#     breed: str | None
#     confidence: float | None

# # Root GET endpoint
# @app.get("/")
# def root():
#     return {"message": "YOLO Breed Predictor API is running! Use POST /predict to send images."}

# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     try:
#         image_path = f"temp_{file.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         results = model.predict(image_path, verbose=False)[0]

#         if results.boxes is not None and len(results.boxes) > 0:
#             breed = results.names[int(results.boxes.cls[0].item())]
#             confidence = round(float(results.boxes.conf[0].item()), 4)
#         elif hasattr(results, "probs") and results.probs is not None:
#             breed = results.names[results.probs.top1]
#             confidence = round(float(results.probs.top1conf), 4)
#         else:
#             breed = None
#             confidence = None

#         os.remove(image_path)
#         return {"breed": breed, "confidence": confidence}

#     except Exception as e:
#         return {"breed": None, "confidence": None}






# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# from ultralytics import YOLO
# import os

# app = FastAPI()

# model = YOLO(r"D:\\cow-breed-final\\cow_breed_yolo11_FineTune.pt")

# # Pydantic model for response
# class PredictResponse(BaseModel):
#     breed: str | None
#     confidence: float | None

# @app.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):
#     try:
#         image_path = f"temp_{file.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         results = model.predict(image_path, verbose=False)[0]

#         if results.boxes is not None and len(results.boxes) > 0:
#             breed = results.names[int(results.boxes.cls[0].item())]
#             confidence = round(float(results.boxes.conf[0].item()), 4)
#         elif hasattr(results, "probs") and results.probs is not None:
#             breed = results.names[results.probs.top1]
#             confidence = round(float(results.probs.top1conf), 4)
#         else:
#             breed = None
#             confidence = None

#         os.remove(image_path)
#         return {"breed": breed, "confidence": confidence}

#     except Exception as e:
#         return {"breed": None, "confidence": None}

























# import os
# from fastapi import FastAPI, File, UploadFile
# from ultralytics import YOLO

# app = FastAPI()
# model = YOLO(r"D:\\cow-breed-final\\cow_breed_yolo11_FineTune.pt")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file temporarily
#         image_path = f"temp_{file.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Run prediction
#         results = model.predict(image_path, verbose=False)[0]

#         # Convert result to JSON-serializable dict
#         result_json = {
#             "boxes": [],
#             "class_names": results.names,
#             "all_class_probs": {},
#         }

#         # Detection boxes
#         if results.boxes is not None:
#             for box in results.boxes:
#                 result_json["boxes"].append({
#                     "cls": int(box.cls.item()),
#                     "breed": results.names[int(box.cls.item())],
#                     "confidence": float(box.conf.item()),
#                     "xyxy": box.xyxy.tolist()
#                 })

#         # Classification probabilities
#         if hasattr(results, "probs") and results.probs is not None:
#             result_json["top1_class"] = results.names[results.probs.top1]
#             result_json["top1_confidence"] = float(results.probs.top1conf)
#             result_json["top5_classes"] = [results.names[i] for i in results.probs.top5]
#             result_json["top5_confidences"] = [float(c) for c in results.probs.top5conf]
#             result_json["all_class_probs"] = {name: float(prob) for name, prob in zip(results.names.values(), results.probs.data)}

#         os.remove(image_path)
#         return result_json

#     except Exception as e:
#         return {"error": str(e)}







# import os
# from fastapi import FastAPI, File, UploadFile
# from ultralytics import YOLO

# app = FastAPI()

# # Load YOLO model once
# model = YOLO(r"D:\\cow-breed-final\\cow_breed_yolo11_FineTune.pt")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file temporarily
#         image_path = f"temp_{file.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Run prediction
#         results = model.predict(image_path, verbose=False)[0]

#         # DEBUG PRINTS: see all model info in console
#         print("=== RESULTS OBJECT ===")
#         print(results)
#         print("Class names:", results.names)

#         all_detections = []
#         all_class_probs = {}
#         top_breed = None
#         top_confidence = None

#         # Detection model (boxes)
#         if results.boxes is not None and len(results.boxes) > 0:
#             print("Boxes detected:")
#             for box in results.boxes:
#                 cls_name = results.names[int(box.cls.item())]
#                 conf = float(box.conf.item())
#                 bbox = box.xyxy.tolist()
#                 print(f"Class: {cls_name}, Confidence: {conf}, BBox: {bbox}")
#                 all_detections.append({"breed": cls_name, "confidence": conf, "bbox": bbox})

#             # Top detection
#             top_breed = results.names[int(results.boxes.cls[0].item())]
#             top_confidence = round(float(results.boxes.conf[0].item()), 4)

#         # Classification model
#         elif hasattr(results, "probs") and results.probs is not None:
#             top_idx = results.probs.top1
#             top_breed = results.names[top_idx]
#             top_confidence = round(float(results.probs.top1conf), 4)

#             print("Classification probabilities:")
#             print("Top1 class:", top_breed, top_confidence)
#             print("Top5 indices:", results.probs.top5)
#             print("Top5 confidences:", results.probs.top5conf)
#             print("All class probabilities:", results.probs.data)

#             all_class_probs = {name: float(prob) for name, prob in zip(results.names.values(), results.probs.data)}

#         # Remove temp file
#         os.remove(image_path)

#         # RETURN everything in JSON
#         return {
#             "top_breed": top_breed,
#             "top_confidence": top_confidence,
#             "all_detections": all_detections,
#             "all_class_probs": all_class_probs
#         }

#     except Exception as e:
#         return {"error": str(e)}











# import os
# from fastapi import FastAPI, File, UploadFile
# from ultralytics import YOLO

# app = FastAPI()

# # Load YOLO model once
# model = YOLO(r"D:\\cow-breed-final\\cow_breed_yolo11_FineTune.pt")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file temporarily
#         image_path = f"temp_{file.filename}"
#         with open(image_path, "wb") as f:
#             f.write(await file.read())

#         # Run prediction
#         results = model.predict(image_path, verbose=False)[0]
#         print(results)

#         # Determine top prediction
#         if results.boxes is not None and len(results.boxes) > 0:
#             # Detection model
#             breed = results.names[int(results.boxes.cls[0].item())]
#             confidence = round(float(results.boxes.conf[0].item()), 2)
#         elif hasattr(results, "probs") and results.probs is not None:
#             # Classification model
#             breed = results.names[results.probs.top1]
#             confidence = round(float(results.probs.top1conf), 2)
#         else:
#             breed = None
#             confidence = None

#         # Remove temp file
#         os.remove(image_path)

#         # Return JSON response
#         return {"breed": breed, "confidence": confidence}

#     except Exception as e:
#         return {"error": str(e)}


