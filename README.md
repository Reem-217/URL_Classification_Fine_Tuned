# 🔐 URL Safety Classifier

This app uses a fine-tuned BERT model to classify URLs as either **Safe** or **Unsafe**.  
Built with 🤗 Hugging Face Transformers and deployed via Gradio.

---

## 🚀 Demo

👉 Try it live: [URL Safety Classifier App](https://huggingface.co/spaces/remomo/URLClassifier)

---

## 🧠 Model

This model is a fine-tuned version of `bert-base-uncased` on a binary classification task for URL safety detection.

- 🤗 Model Hub: [remomo/URLClassifer](https://huggingface.co/remomo/URLClassifier)

---

## 📊 Training Results

| Epoch | Training Loss | Validation Loss | Accuracy | AUC    |
|-------|---------------|------------------|----------|--------|
| 1     | 0.5044        | 0.3851           | 81.6%    | 0.914  |
| 2     | 0.4071        | 0.3333           | 85.1%    | 0.932  |
| 3     | 0.3553        | 0.3120           | 85.6%    | 0.940  |
| 4     | 0.3584        | 0.3512           | 84.4%    | 0.946  |
| 5     | 0.3521        | 0.3400           | 86.7%    | 0.948  |
| 6     | 0.3475        | 0.2895           | 87.1%    | 0.951  |
| 7     | 0.3351        | 0.2875           | 87.6%    | 0.950  |
| 8     | 0.3117        | 0.2890           | 86.9%    | 0.950  |
| 9     | 0.3145        | 0.2841           | 87.3%    | 0.951  |
| 10    | 0.3134        | 0.2889           | 86.7%    | 0.951  |

📉 **Final Training Loss**: `0.3599`  
📈 **Best AUC**: `0.951`

---

## 📁 Dataset

Used a labeled dataset of URLs classified as Safe or Unsafe.

- 🔗 Dataset: [URL Classification Dataset](shawhin/phishing-site-classification)

---

## 🛠️ Tech Stack

- 🤗 Transformers
- 🐍 PyTorch
- 💬 Gradio
- Hugging Face Hub & Spaces

---

## 🧪 Usage

You can use this model in Python:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("your-username/url-safety-bert")
tokenizer = AutoTokenizer.from_pretrained("your-username/url-safety-bert")

inputs = tokenizer("http://example.com", return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=1).item()
