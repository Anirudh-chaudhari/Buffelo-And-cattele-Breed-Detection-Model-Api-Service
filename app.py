from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import os

app = FastAPI()

# ✅ CORS setup
allowed_origins = [
    "http://localhost:3000",        # Next.js local dev
    "http://127.0.0.1:5500",        # plain HTML/JS dev (if you use it)
    "https://your-site.vercel.app", # replace later with your real domain
    "https://your-site.netlify.app" # replace later if using Netlify
    # You can temporarily use "*" here during dev, but not in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # or ["*"] during testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO(r"cow_breed_yolo11_FineTune.pt")

# Pydantic model for response
class PredictResponse(BaseModel):
    breed: str | None
    confidence: float | None

# Root GET endpoint
@app.get("/")
def root():
    return {"message": "YOLO Breed Predictor API is running! Use POST /predict to send images."}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        results = model.predict(image_path, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            breed = results.names[int(results.boxes.cls[0].item())]
            confidence = round(float(results.boxes.conf[0].item()), 4)
        elif hasattr(results, "probs") and results.probs is not None:
            breed = results.names[results.probs.top1]
            confidence = round(float(results.probs.top1conf), 4)
        else:
            breed = None
            confidence = None

        os.remove(image_path)
        return {"breed": breed, "confidence": confidence}

    except Exception as e:
        return {"breed": None, "confidence": None}










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


