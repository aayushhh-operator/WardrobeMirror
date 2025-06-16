import os, base64, requests, pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ENDPOINT = "product-fashion-matching-02/2"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load precomputed features
with open("models/similarity_model.pkl", "rb") as f:
    data = pickle.load(f)
features_np = data["features"]
product_ids = data["product_ids"]
df = pd.read_csv("data/ready_dataset.csv")

def crop_with_bbox(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    url = f"https://detect.roboflow.com/{MODEL_ENDPOINT}"
    params = {"api_key": API_KEY}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    r = requests.post(url, params=params, data=img_b64, headers=headers)
    r.raise_for_status()
    preds = r.json().get("predictions", [])
    if not preds:
        return None

    p = preds[0]
    x, y, w, h = p['x'], p['y'], p['width'], p['height']
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    img = Image.open(image_path).convert("RGB")
    return img.crop((x1, y1, x2, y2))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        if f:
            path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
            f.save(path)

            cropped = crop_with_bbox(path)
            if cropped is None:
                return render_template("index.html", error="No bounding box detected.")

            tensor = transform(cropped).unsqueeze(0).to(device)
            with torch.no_grad():
                qf = resnet(tensor).squeeze().cpu().numpy()

            sims = cosine_similarity([qf], features_np)[0]
            top_idxs = sims.argsort()[-10:][::-1]
            top_pids = [product_ids[i] for i in top_idxs]

            res = df[df.product_id.isin(top_pids)][["product_name", "feature_image_s3"]].drop_duplicates()
            results = res.to_dict("records")
            return render_template("index.html", uploaded_image=path, results=results)

    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
