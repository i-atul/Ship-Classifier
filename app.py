from flask import Flask, render_template, request
from src.pipeline.prediction import PredictionPipeline
from pathlib import Path
from src.entity.config_entity import DeploymentConfig
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction='No file part')
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        if file:
            # Save the uploaded file to a temp location
            upload_folder = 'uploads'
            os.makedirs(upload_folder, exist_ok=True)
            file_path = Path(upload_folder) / file.filename
            file.save(str(file_path))
            # Run prediction
            predictor = PredictionPipeline(str(file_path))
            result = predictor.predict()
            prediction = result['image']
            os.remove(file_path)  # Optionally remove the file after prediction
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=DeploymentConfig.debug, host=DeploymentConfig.app_host, port=DeploymentConfig.app_port)
