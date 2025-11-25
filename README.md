# Sign2Text: Real-Time Sign Language Recognition System

🧏 **Sign2Text: Real-Time Sign Language Recognition System**\
A web-based computer vision project that translates hand signs into text
subtitles in real time.

------------------------------------------------------------------------

## 📋 Table of Contents

-   💡 Project Overview\
-   ✨ Key Features\
-   🛠️ Technical Stack\
-   🚀 Installation and Setup\
-   💻 Usage
    -   1.  Model Training\

    -   2.  Running the Web Application\
-   📊 Machine Learning Pipeline\
-   🗓️ Project Timeline Summary\
-   👥 Team Members

------------------------------------------------------------------------

## 💡 Project Overview

Sign2Text is a computer vision and machine learning project dedicated to
bridging the communication gap by converting American Sign Language
(ASL) hand signs into written text subtitles.

The final deliverable is a real-time, web-based application that
utilizes the user's webcam to capture hand signs, preprocess the images
to extract key features, and pass these features to a trained Machine
Learning model for letter recognition. The recognized letters are then
combined into coherent words and sentences, which appear directly on the
web interface.

This system was developed using a manually collected dataset to solve
challenges like prediction stability and overfitting.

------------------------------------------------------------------------

## ✨ Key Features

-   **Real-Time Recognition:** Live sign language translation using a
    standard webcam.\
-   **Web-Based Interface:** Built with Flask-SocketIO for low-latency
    prediction streaming.\
-   **Custom Data-Driven Model:** Random Forest classifier trained on
    custom-collected data.\
-   **Robust Preprocessing:** MediaPipe Hands extracts normalized
    landmarks for consistent input.\
-   **Prediction Stabilization:** Smoothing and queue logic reduce
    flicker and stabilize predictions.

------------------------------------------------------------------------

## 🛠️ Technical Stack

  -----------------------------------------------------------------------
  Category           Tool / Library                    Purpose
  ------------------ --------------------------------- ------------------
  Language           Python (3.x)                      Core programming
                                                       language

  Web Framework      Flask, Flask-SocketIO, Eventlet   Backend server &
                                                       real-time
                                                       communication

  Computer Vision    OpenCV (cv2)                      Video handling &
                                                       frame processing

  Feature Extraction MediaPipe Hands                   Landmark
                                                       extraction

  Machine Learning   Scikit-learn                      Predicting hand
                     (RandomForestClassifier)          signs

  Data & Math        NumPy, Pandas                     Data manipulation

  Development        Git, GitHub, Jupyter Notebook     Version control &
                                                       experimentation
  -----------------------------------------------------------------------

The project originally planned to use TensorFlow/Keras, but the final
model uses Scikit-learn's Random Forest classifier
(`model_rfFINAAAAAL.p`).

------------------------------------------------------------------------

## 🚀 Installation and Setup

### Prerequisites

-   Python 3.8+
-   Webcam

### Steps

#### 1. Clone the Repository:

``` bash
git clone [Your-GitHub-Repo-Link]
cd Sign2Text
```

#### 2. Install Dependencies:

``` bash
pip install -r requirements.txt
```

#### 3. Verify Model and Data:

Ensure these files exist in the root directory: - `data_merged.pickle` -
`model_rfFINAAAAAL.p`

------------------------------------------------------------------------

## 💻 Usage

The project supports two workflows: **Model Training** and **Running the
Web Application**.

------------------------------------------------------------------------

### 1. Model Training

  -----------------------------------------------------------------------------------------------
  Step         File                    Description                 Command
  ------------ ----------------------- --------------------------- ------------------------------
  1            `collect_imgs.py`       Capture images for each     `python collect_imgs.py`
                                       class (letter)              

  2            `preprocess.py`         Extract MediaPipe landmarks `python preprocess.py`
                                       → pickle file               

  3            `merge_pickles.py`      Merge multiple datasets     `python merge_pickles.py`

  4            `train_classifier.py`   Train Random Forest model   `python train_classifier.py`
  -----------------------------------------------------------------------------------------------

------------------------------------------------------------------------

### 2. Running the Web Application

#### Start the Server:

``` bash
python app2.py
```

#### Open the App:

Visit the local server link shown in the console (usually
`http://127.0.0.1:5000`).

------------------------------------------------------------------------

## 📊 Machine Learning Pipeline

1.  **Data Collection:** Webcam captures images for each sign.\
2.  **Feature Extraction:** MediaPipe extracts 21 hand landmarks →
    normalized into 42 features.\
3.  **Data Preparation:** All samples merged into `data_merged.pickle`.\
4.  **Model Training:** Random Forest trained to classify signs.\
5.  **Inference:** The trained model runs inside a Flask-SocketIO app to
    perform real-time recognition.

------------------------------------------------------------------------

## 🗓️ Project Timeline Summary

  Weeks   Deliverables                                      Focus
  ------- ------------------------------------------------- ----------------
  1--2    Environment setup, preprocessing tests, metrics   Setup & Data
  3--4    CNN prototype, webcam demo                        Initial Model
  5--6    Static web page                                   Web Dev
  7--8    Improved model, stabilization, bug fixes          Optimization
  9--10   Custom dataset collection, final deployment       Final Delivery

------------------------------------------------------------------------

## 👥 Team Members

  Role          Name
  ------------- ---------------------
  Team Leader   Zeinab Osama
  Team Member   Hana Ibrahim
  Team Member   Abdelrahman Darwish
  Team Member   Ahmed Saeed
  Team Member   Amr Fouad
  Team Member   Mahmoud Hallul

------------------------------------------------------------------------
