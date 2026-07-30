# Image Generation

## 📌 Description

An end-to-end computer vision project for text-to-image generation and image analysis. This repository features data exploration, model experimentation, and an interactive Streamlit dashboard. Users can generate creative images from text prompts, analyze visual content, and perform prompt engineering experiments using advanced Generative AI and deep learning pipelines.

---

## 🛠️ Tech Stack

| Category                    | Technologies Used                                                                                                                      |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| 🌐 **Programming Language** | `Python`                                                                                                                               |
| 🌱 **Environment**          | `Jupyter Notebook`                                                                                                                     |
| 🧩 **Frameworks**           | `PyTorch`, `Streamlit`                                                                                                                 |
| ⚛️ **Libraries**            | `NumPy`, `Matplotlib`, `Diffusers`, `pyngrok`, `Streamlit - Drawable Canvas`,<br>`Transformers`, `Pillow`, `Accelerate`, `safetensors` |
| ⚡ **Tool**                 | `Google Colab`                                                                                                                         |
| 🚧 **Tunneling Service**    | `ngrok`                                                                                                                                |

---

## ⚙️ Setup Instructions

1. **Prerequisites**
   - Python 3.11 or higher.
   - Git installed on your system.
   - An active [ngrok](https://ngrok.com/) account.

2. **Ngrok Authtoken Setup**
   - Visit the official [ngrok website](https://ngrok.com/).
   - Sign up for a new account or log in to your existing account.
   - Once redirected to the ngrok dashboard, navigate to the **Your Authtoken** menu on the left sidebar.
   - Copy your unique authentication token to use during the configuration phase.

3. **Clone the Repository**

```bash
git clone https://github.com/Fikri-Rouzan/image-generation.git
cd image-generation
```

4. **Configure Authentication Token**

   Open the `streamlit.ipynb` file and insert your ngrok authtoken into the following code cell

   ```python
   auth_token = "YOUR_AUTHENTICATION_KEY"

   ngrok.set_auth_token(auth_token)
   subprocess.Popen(["streamlit", "run", "app.py"])
   ```
