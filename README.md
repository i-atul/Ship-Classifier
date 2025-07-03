# Ship-Classifier

A robust, production-ready deep learning project for multi-class ships images classification using PyTorch, DVC, and Docker. Built for reproducibility, scalability, and easy deployment.

## Project Workflow

1. **Data Ingestion**: Downloads and unpacks ship image datasets.
2. **Preprocessing**: Cleans and prepares data for training.
3. **Model Training**: Trains a PyTorch model for multi-class ship classification.
4. **Model Evaluation**: Evaluates model performance on validation data.
5. **Model Deployment**: Serves predictions via a Flask API.
6. **Experiment Tracking**: Uses DVC and MLflow for versioning and experiment management.

## Supported Ship Classes
- Cargo
- Tanker
- Passenger
- Fishing
- Military
- Other (customizable)

## Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/your-username/Ship-Classifier.git
cd Ship-Classifier
```

### 2. Install Requirements (for local development)
```sh
pip install -r requirements.txt
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
docker build -t as135/ship-classifier .
```

### Pull from Docker Hub
```sh
docker pull as135/ship-classifier:latest
```

### Run with Docker
```sh
docker run -p 5000:5000 as135/ship-classifier
```

### Run with Docker Compose
```sh
docker compose up -d
```

## Model Artifacts
- Trained models are stored in `artifacts/training/model.h5` (or `.pth` for PyTorch).
- The Docker image includes the trained model for inference.
- You can mount your own model with Docker Compose using:
  ```yaml
  volumes:
    - ./artifacts:/app/artifacts
  ```

## Project Structure
```
├── app.py                # Flask API for prediction
├── Dockerfile            # Docker build instructions
├── docker-compose.yml    # Multi-container orchestration
├── dvc.yaml              # DVC pipeline stages
├── requirements.txt      # Python dependencies
├── src/                  # Source code (pipelines, components, utils)
├── artifacts/            # Model artifacts and data
└── ...
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