import os
os.environ['CURL_CA_BUNDLE'] = ''
from app_folder import create_app
import webbrowser
import threading
import time

app = create_app()

def open_browser():
    # Wait for the Flask app to start
    time.sleep(1)  # Adjust the delay if needed
    # Open the browser to the Flask app's URL
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Start the browser opening in a separate thread
    threading.Thread(target=open_browser).start()
    # Run the Flask app
    app.run(debug=True)
