Instructions
Prerequisites:
- Python 3.10
- Visual Studio Code
Steps to Run the Project
1. Clone the Repository: git clone < GitHub> cd image_retrieval_system
2. Set Up a Virtual Environment: python -m venv
3. Activate the Virtual Environment:
- On Windows:
• \venv\Scripts\activate
4. Install Dependencies:
pip install -r requirements.txt
5. Add Images to the Dataset:
- Place your images in the
•static/Images/
folder.
6. Run
the Application:
python run. py
7. Access the Application:
- Open a browser and navigate to http://127.0.0.1:5000.
â€¢ Text-Based Search:
1. Enter a text query in the search box
2. Click Search.
3. The system will display the top 5 most similar images.
â€¢ Voice-Based Search:
1. Click the Voice Search button.
2. Speak your query into the microphone (e.g.,
"show me a beach').
Configuration
The application can be configured by modifying the
"app/config-py file. Key configurations include:
- IMAGE FOLDER: Path to the folder containing images for retrieval.
Dependencies
The project requires the following Python packages (listed in
requirements.txt ):
Flask==2.3.2
torch==2.0.1
transformers=-4.30.2
Pillow==9.5.0 numpy==1.24.3
scikit-learn==1.2.2
SpeechRecoanition==3.10.0
pyttsx3 == 2.90

