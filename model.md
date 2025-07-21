## ✅ **1. Download the model from Google Drive in Colab**

The link:

```
https://drive.google.com/file/d/194J7lFyjy-Vp9LknoJr99_y9jSLNT5X4/view?usp=sharing
```

The **FILE\_ID** is:

```
194J7lFyjy-Vp9LknoJr99_y9jSLNT5X4
```

Use this code:

```python
import gdown

file_id = "194J7lFyjy-Vp9LknoJr99_y9jSLNT5X4"
url = f"https://drive.google.com/uc?id={file_id}"
output = "age_model_resnet50.h5"

gdown.download(url, output, quiet=False)
print("Model downloaded successfully!")
```

---

## ✅ **2. Load the pre-trained model**

```python
from tensorflow.keras.models import load_model

age_model = load_model("age_model_resnet50.h5")
print("Model loaded successfully and ready for inference!")
```

---

## ✅ **3. Test a prediction**

```python
pred = predict_age("path_to_test_image.jpg")
print("Predicted Age:", pred)
```



