# Image Retrieval System

## Prerequisites

Ensure you have the following installed before running the project:

- **Python**: 3.10
- **Visual Studio Code**

## Steps to Run the Project

### 1. Clone the Repository
```sh
git clone https://github.com/ASIVEMNQUMENI/Image-Retrieval.git
cd image_retrieval_system
```

### 2. Set Up a Virtual Environment
open the terminal
```sh
python -m venv venv
```

### 3. Activate the Virtual Environment
- **On Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```sh
  source venv/bin/activate
  ```

### 4. Install Dependencies
```sh
pip install -r requirements.txt
pip install python-certifi-win32
```

### 5. Add Images to the Dataset
Place your images in the following folder:
```
/static/Images/
```

### 6. Run the Application
```sh
python run.py
```

---

## Features

### **Text-Based Search**
1. Enter a text query in the search box.
2. Click **Search**.
3. The system will display the **top 5 most similar images**.

### **Voice-Based Search**
1. Click the **Voice Search** button.
2. Speak your query into the microphone (e.g., *"Show me a beach"*).
3. The system will retrieve and display the most relevant images.

---

## Configuration
The application can be configured by modifying the following file:
```
app/config.py
```
Key configurations include:
- **IMAGE_FOLDER**: Path to the folder containing images for retrieval.

---

## Dependencies
The project requires the following Python packages (listed in `requirements.txt`):
```txt
Flask==2.3.2
torch==2.0.1
transformers==4.30.2
Pillow==9.5.0
numpy==1.24.3
scikit-learn==1.2.2
SpeechRecognition==3.10.0
pyttsx3==2.90
```

---


