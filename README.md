# Ship-Classifier

A robust, production-ready deep learning project for multi-class ships images classification using PyTorch, Built for reproducibility, scalability, and easy deployment.

## Project Workflow

1. **Data Ingestion**: Downloads and unpacks ship image datasets.
2. **Preprocessing**: Cleans and prepares data for training.
3. **Prepare Base Model**: Loads a pre-trained VGG16 model.
4. **Model Training**: Trains a PyTorch model for multi-class ship classification.
5. **Model Evaluation**: Evaluates model performance on validation data.
6. **Model Deployment**: Serves predictions via a Flask App.
7. **Experiment Tracking**: Uses DVC and MLflow for versioning and experiment management.
8. **CI/CD Integration**: Implements continuous integration and deployment using GitHub Actions for automated testing, building, and deployment.

## Supported Ship Classes
- Cargo
- Tanker
- Cruise


## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/i-atul/Ships-Image-Classifier.git
cd Ships-Image-Classifier
```

### 2. Install Requirements (for local development)
```sh
pip install -r requirements.txt
pip install -e .
```

### 3. Reproduce the Pipeline with DVC
```sh
dvc repro
```

### 4. Train the Model
```sh
python src/pipeline/training.py
```

### 5. Run the Flask App (Locally)
```sh
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) to use the web interface.

## Docker Usage

### Build the Docker Image
```sh
docker build -t <username>/<imagename> .
```

### Pull from Docker Hub
```sh
docker pull as135/ship-classifier
```

### Run with Docker
```sh
docker run -p 5000:5000 <username>/<imagename>
```

### Run with Docker Compose
```sh
docker compose up -d
```


## Reproducibility & MLOps
- **DVC**: Data and model versioning
- **Docker**: Consistent, portable environments
- **MLflow**: Experiment tracking (optional)
- **CI/CD Ready**: Easily integrate with cloud or on-prem pipelines

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

## License
This project is licensed under the MIT License.
