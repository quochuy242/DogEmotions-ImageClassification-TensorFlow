# Emotion Classification with Deep Learning Models

This project aims to classify emotions from facial images using deep learning techniques implemented in TensorFlow. The trained model is then deployed as a web application using Flask and containerized with Docker for easy deployment and scalability.

## Project Overview

The core of this project is a deep learning model built with **TensorFlow** that can classify facial expressions into various emotion categories such as *happiness, sadness, anger, fear, neutral and surprise*. The model is trained on a labeled dataset of facial images (**FER2013**), allowing it to learn the intricate patterns and features associated with different emotions.

To make the model accessible and usable, it is deployed as a web application using **Flask**, a lightweight Python web framework. This application provides a user-friendly interface where users can upload facial images, and the deep learning model will analyze them and return the predicted emotion.

Furthermore, the entire application is containerized using **Docker**, ensuring consistent and reproducible deployments across different environments. This containerization approach simplifies the deployment process and allows for easy scaling and management of the application.

## Features

- Deep learning model for emotion classification built with TensorFlow
- Trained on a labeled dataset of facial expressions
- Web application interface built with Flask
- User-friendly interface for uploading facial images
- Emotion prediction results displayed on the web page
- Docker containerization for easy deployment and scalability

## Getting Started

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/quochuy242/Emotions-ImageClassification.git`

2. Navigate to the project directory: `cd Emotions-ImageClassification`
3. Build the Docker image: `docker build -t Emotions-ImageClassification .`
4. Run the Docker container: `docker run -p 5000:5000 Emotions-ImageClassification`
5. Access the web application at `http://localhost:5000`

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
