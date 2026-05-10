# MoodDJ-Music-Mood-Classifier-and-Recommender
Machine Learning 2 Group project assignment

[README.md](https://github.com/user-attachments/files/27571176/README.md)
# Mood DJ - Music Mood Classifier and Recommender

**Created by**: Adib Mukhriz Abdul Malek, Almási Júlia, Bartal Dávid, and Kiss Boróka

## Demo
![Demo](demo.png)
![Recommended songs](demo2.png)

## Overview
Mood DJ is a deep learning-based music mood classification and recommendation system built using PyTorch, Librosa, and Gradio. The project analyzes uploaded music tracks using MFCC audio features and predicts their emotional mood using a Convolutional Neural Network (CNN).

Based on the predicted emotion, the system recommends similar songs from a preprocessed music library using cosine similarity. The application includes an interactive Gradio interface where users can upload a song, receive a mood prediction, and listen to recommended tracks.

The project focuses on combining audio signal processing, deep learning, and content-based recommendation into a simple end-to-end music recommendation workflow.

## Features
- Music mood classification using CNNs
- MFCC-based audio feature extraction
- Content-based song recommendation
- Interactive Gradio web interface
- Comparison with Random Forest and SVM baselines

## Technical architecture

1.  **Feature extraction**: features are extracted from audio files using the `librosa` package by computing MFCC representations. A random 5-second sample is taken from the songs to save processing time.
2.  **Model:** The classification model consists of three convolutional layers combined with Batch Normalization, Global Average Pooling, and Dropout regularization.
3.  **Recommendation UI**: The recommendation UI is built using the `gradio` package. The UI allows users to upload any MP3 audio file, view its mood, and receive 5 song recommendations from the pre-trained dataset. You can also listen to the top recommendation in the UI.
4.  **Recommendation logic**: The system uses **Cosine Similarity** to compare the mean MFCC vector of the user's uploaded clip against the pre-computed feature cache. It only recommends songs that share the same predicted emotion label as the uploaded track.

## Dataset

This project uses the Kaggle **Audio-Emotion-Context Dataset (DEAM)**:

Source: https://www.kaggle.com/datasets/ziya07/audio-emotion-context-dataset

The dataset contains music tracks annotated with emotional labels and contextual metadata, making it suitable for our supervised music emotion classification task.

- Only tracks with valid emotion labels were used
- Audio files were processed into MFCC representations
- The final dataset consisted of 1546 usable audio samples
- Four emotion classes were used:
  - Angry
  - Happy
  - Relaxed
  - Sad

To reduce computational cost, random 5-second segments were sampled from each audio track during preprocessing.


## Workflow
1. Audio files are converted into MFCC representations using `librosa`.
2. MFCC matrices are padded/truncated into a fixed `(215, 40)` shape.
3. A CNN classifies songs into four emotion labels:
   - Angry
   - Happy
   - Relaxed
   - Sad
4. Songs with the same predicted emotion are compared using cosine similarity.
5. The top 5 closest matches are returned through a Gradio interface.

## Usage

### Requirements

The project is designed to run on Google Colab with T4 GPU support.

#### Tech Stack
- PyTorch
- Librosa
- Gradio
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install librosa torch gradio scikit-learn numpy pandas matplotlib
```

The `archive.zip` folder from the dataset should be placed in the `/MyDrive/MachineLearning/` for the songs to be loaded.


### Running

The notebook is to be executed from start to bottom. The last cell in the notebook initializes the Gradio UI, which can be used either inline or on a publicly accessible URL.

The notebook is written with a cache-first logic in mind to speed up learning and execution. MFCC's are precomputed for the audio tracks and loaded from `.npy` files if they exist in the `/MyDrive/MachineLearning/` directory. The first execution will take significantly longer (approx. 10-15 minutes) as it must extract MFCCs for 1,546 audio files. Subsequent runs use computed MFCC features from the Drive folder.

## Performance
Due to limited labeled samples, the dataset size constrained the overall classification accuracy of the model.

Our model currently achieves an accuracy of approximately 24%. This performance is comparable to the Random Forest and SVM baselines, highlighting the limitations imposed by dataset size and short audio segments.

| Model | Accuracy |
|---|---|
| CNN | 25.484% |
| Random Forest | 23.548% |
| SVM | 27.419% |

## Future improvements

The project can be further improved by including other features from audio, such as Chroma features (for harmony) and Spectral Centroid (for brightness). Increasing dataset size and integrating Spotify API could also improve the performance. 

## References

-   **Librosa:** Mcfee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python.

-   **Dataset:** Kaggle "Audio-Emotion-Context-Dataset".


This project demonstrates how deep learning and audio feature extraction can be combined to build an end-to-end music mood recommendation system.
