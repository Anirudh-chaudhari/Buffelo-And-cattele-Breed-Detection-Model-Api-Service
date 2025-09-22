
# 🐄 Cattle & Buffalo Breed Recognition API

This project provides a **FastAPI-based backend** for cattle & buffalo breed recognition in India. It uses a **YOLOv11 model fine-tuned on Indian bovine datasets** to predict the breed of cattle/buffalo from an uploaded image.

The API connects with a **Next.js + Tailwind frontend**, and can be deployed using **Render (backend)** + **Netlify/Vercel (frontend)**.

---

## 🚀 Features

* Upload an image of a cow/buffalo → get **predicted breed** + **confidence score**
* **YOLOv11 model** trained on Indian breeds
* Built with **FastAPI** for speed and scalability
* **CORS enabled** for browser → API integration
* Ready for **cloud deployment** (Render, Dockerized if needed)

---

## 📂 Project Structure

```
cow-breed-api/
│
├── app.py                        # FastAPI application (this file)
├── cow_breed_yolo11_FineTune.pt  # Trained YOLOv11 model weights
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/cow-breed-api.git
   cd cow-breed-api
   ```

2. Create a virtual environment & install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

   pip install -r requirements.txt
   ```

3. Place the YOLOv11 model weights file:

   ```
   cow_breed_yolo11_FineTune.pt
   ```

---

## ▶️ Running Locally

Start FastAPI server with **Uvicorn**:

```bash
uvicorn app:app --reload --port 8000
```

API will be available at:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📡 API Endpoints

### Root Endpoint

**GET /**

```json
{
  "message": "YOLO Breed Predictor API is running! Use POST /predict to send images."
}
```

### Predict Endpoint

**POST /predict**

* Accepts: Image file (`multipart/form-data`)
* Returns: Predicted **breed** + **confidence score**

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test_buffalo.jpg"
```

Example response:

```json
{
  "breed": "Jaffrabadi",
  "confidence": 0.9931
}
```

---

## 🌐 CORS Setup

The backend allows requests from **localhost** (development) and **Netlify/Vercel** domains (deployment):

```python
allowed_origins = [
    "http://localhost:3000",        # Next.js local dev
    "http://127.0.0.1:5500",        # Plain HTML/JS dev
    "https://your-site.vercel.app", # Replace with deployed domain
    "https://your-site.netlify.app"
]
```

During development, use `"*"` for testing, but restrict in production.

---

## 🔗 End-to-End Workflow

### Data Preprocessing & Dataset Split

* **70% Training** → Train YOLOv11 model
* **15% Validation** → Fine-tune, avoid overfitting
* **15% Testing** → Evaluate accuracy
* Steps: resizing, augmentation, cleaning, balancing

### Model Training Workflow

* Dataset → YOLOv11 Training → Validation → Accuracy Evaluation → Optimized Model
* Iterative training with epochs (loss decreases, accuracy improves)

### Model Deployment (Backend)

* Trained Model → FastAPI Backend → Hosted on Render (API Endpoint)
* Output: JSON `{breed, confidence}`

### Connecting Frontend & Backend

* Website (Next.js + Tailwind) → API Request (CORS enabled) → FastAPI Backend → YOLOv11 Model → JSON Response → Result Displayed
* Example output: *"Breed: Jaffrabadi, Confidence: 99.3%"*

### Publishing Website

* Localhost (development) → Deploy frontend on **Netlify/Vercel** → Connect to Render API → Live Website 🌐
* Farmer workflow: Upload → Prediction → Result

---

## 🛠️ Tech Stack

* **Model**: YOLOv11 (Ultralytics)
* **Backend**: FastAPI, Uvicorn, Pydantic
* **Frontend**: Next.js, TailwindCSS
* **Deployment**: Render (backend), Netlify/Vercel (frontend)

---

## 📌 Example Workflow

👨‍🌾 Farmer uploads cow/buffalo image →
🌐 Website (Next.js frontend) sends request →
⚡ FastAPI backend (Render) runs YOLOv11 →
📊 Returns JSON `{breed, confidence}` →
✅ Result displayed on website
