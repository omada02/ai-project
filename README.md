# AI Project - FashionMNIST Classifier

**Matteo Adamo - VR471358**

## 🚀 Come eseguire

### 1. Clona la repo
```bash
git clone https://github.com/omada02/ai-project.git
cd ai-project/AI


### 2. Crea ed esegui (senza Docker)
```bash
pip install -r requirements.txt
pip install -e .
python src/train.py
python app/gradio_app.py
```

### 3. Oppure con Docker
```bash
docker build -t fashionmnist-app .
docker run -p 7860:7860 fashionmnist-app
```

### 4. EDA
Apri notebooks/eda.ipynb con Jupyter Notebook o VSCode.

### 5. CI/CD
Push su GitHub attiverà test + lint via GitHub Actions

## 📤 Docker Hub
Configura `docker login`, poi:
```bash
docker tag fashionmnist-app matteoadamo/fashionmnist-app
docker push matteoadamo/fashionmnist-app
```

## 🧠 Modello
Modello MLP in PyTorch per classificare immagini FashionMNIST (28x28, 10 classi).
