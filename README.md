# Spotify Track Analytics Popularity Prediction

## Content
* [Readme.md](https://github.com/YShutko/CI_spotify_track_analysis/blob/210b32764e51e58e27a977b8c2a8e6f54010f5ba/README.md)
* [Datasets](https://github.com/YShutko/CI_spotify_track_analysis/tree/210b32764e51e58e27a977b8c2a8e6f54010f5ba/data) 
* [Jupyter notebooks](https://github.com/YShutko/CI_spotify_track_analysis/tree/master/notebooks)

## Dataset Content
The data set used for this project: [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). The collection of ~114,000 songs across 125 genres with features like danceability, energy, tempo, and popularity. Ideal for audio analysis, genre classification, and music trend exploration.

The dataset consists of the following columns:
* track_id: Unique Spotify identifier for each track.
* artists: List of artists performing the track, separated by semicolons.
* album_name: Title of the album where the track appears.
* track_name: Title of the song.
* popularity: Score from 0–100 based on recent play counts; higher means more popular.
* duration_ms: Length of the track in milliseconds.
* explicit: Indicates whether the track contains explicit content (True/False).
* danceability: Score (0.0–1.0) measuring how suitable the song is for dancing.
* energy: Score (0.0–1.0) reflecting intensity, speed, and loudness.
* key: Musical key using Pitch Class notation (0 = C, 1 = C♯/D♭, etc.).
* loudness: Overall volume of the track in decibels.
* mode: Indicates scale type (1 = major, 0 = minor).
* speechiness: Score estimating spoken content in the track.
* cousticness: Likelihood (0.0–1.0) that the song is acoustic.
* instrumentalness: Probability that the track has no vocals.
* liveness: Measures if the song was recorded live (higher = more live).
* valence: Positivity of the music (0.0 = sad, 1.0 = happy).
* tempo: Speed of the song in beats per minute (BPM).
time_signature: Musical meter (e.g. 4 = 4/4 time).
track_genre: Musical genre classification of the track.

## Business Problem and Data Science Objectives overview

The Spotify dataset contains detailed track-level metadata, including musical attributes like energy, valence (positivity), tempo, danceability, loudness, and genre. The goal is to leverage this data to:
* Understand what makes a song popular
* Enable mood-based playlist curation
* Build predictive and recommendation tools for artists and curators.
  
## Business Requirements
1. Understand Key Drivers of Song Popularity
Objective: Stakeholders want to identify which musical and structural features (e.g., energy, valence, tempo, duration, genre) most strongly influence a track’s popularity score.  
Approach:  
* Perform exploratory data analysis (EDA) to study correlations between features and popularity.
* Build feature importance models (e.g., Random Forest Regressor, XGBoost).
* Visualize top predictive features to communicate key insights.
Deliverables:
* “Top 10 Features Influencing Popularity” chart
* Feature importance report with actionable insights (e.g., “Energy and Valence increase popularity by +12 points”).

2️. Classify Songs by Mood and Energy
Objective: The product team needs a system to categorize songs by mood (e.g., happy, sad, energetic, calm) for playlist generation and UX personalization.
Approach:
* Use valence (positivity) and energy as key emotional indicators.
* Create mood labels using thresholds or unsupervised clustering:
* High valence + high energy → Happy/Energetic
* Low valence + low energy → Calm/Melancholic
* Train a mood classification model (Logistic Regression or Random Forest).
Deliverables:
* Mood classification categories and definitions
* Interactive “Mood Map” visualization of tracks by valence/energy

3️. Genre-Level Analysis
Objective: Provide the marketing and analytics teams with insights into genre performance and audience preferences.
Approach:
* Aggregate and visualize average popularity by genre
* Identify top and emerging genres
* Use clustering to reveal genre subgroups with similar sound characteristics
Deliverables:
* Genre trend dashboards (avg. popularity, tempo, valence)
* Heatmaps comparing genre vs. energy and mood

4️. Support Playlist Curation
Objective: Help users and curators find and group similar tracks automatically for playlist creation.
Approach:
* Use K-Means clustering or cosine similarity on normalized audio features (energy, danceability, tempo, etc.)
* Build a “Find Similar Songs” function that recommends tracks closest in feature space.
Deliverables:
* “Song Similarity Map” visualization
* Playlist recommendation demo

5️. Data-Driven Music Recommendations
Objective: Enable producers and managers to make data-informed creative decisions before releasing songs.
Approach:
* Train a regression model to predict popularity score (0–100) for new songs.
* Generate actionable insights such as:
* Predicted popularity
* Feature-level improvement suggestions (e.g., +10 energy = +6 popularity points)
* Benchmarking against top songs in the same genre
Deliverables:
* AI-powered “Track Optimizer” dashboard
* Predictive model for new song success


## Hypothesis


## Project Plan
* Data Acquisition & Preparation
  * Load and explore the Spotify tracks dataset from Kaggle.
  * Clean and preprocess data: handle missing values, convert duration to minutes, normalize/scale relevant features.
* Exploratory Data Analysis (EDA)
  * Analyze distribution of popularity, tempo, valence, energy, and other audio features.
  * Visualize relationships between key features using:
      * Correlation heatmaps
      * Pairplots and histograms
* Interactive Dashboards & Gradio Interface
  * Create an interactive Gradio interface that allows users to:
  * Upload new track features and get popularity predictions
  *  Explore how changes in tempo, energy, and valence affect classification
  * Visualize real-time audio feature comparisons across genres or user inputs
    
## The rationale to map the business requirements to the Data Visualisations


## Dashboard Design
  
## Analysis techniques used
* Visual Studio Code
* Python
* Jupyter notebook
* ChatGPT

## Ethical consideration
This project utilizes publicly available Spotify track data for the purpose of educational data analysis and machine learning model development. The following ethical aspects were taken into account:
* **Data Privacy:**  
  The dataset does not include any personal or user-specific information. All data refers to music track features and metadata that are publicly accessible through Spotify APIs. No individual listeners are identified or targeted.
* **Copyright and Usage Rights:**  
  Although track metadata (e.g., names, artists, albums) is included, no copyrighted audio content is used. The dataset complies with fair use for academic and non-commercial research purposes.
* **Bias and Fairness:**  
  The dataset may reflect biases in genre popularity or artist representation due to Spotify’s algorithmic and user-driven metrics. These biases can affect model predictions and should be acknowledged in analysis or applications.
* **Responsible Use of Predictions:**  
  Any popularity or mood predictions made using this data should not be used to marginalize artists or genres. The results represent trends in the data, not objective quality judgments.
* **Transparency:**  
  All analysis steps, assumptions, and model performance evaluations are documented for transparency. This promotes responsible AI usage and allows users to understand how conclusions were derived.
  
## Development Roadmap

## Planning:
* GitHub [Project Board](https://github.com/users/YShutko/projects/6) was used to plan and track the progress.

## Main Data Analysis Libraries
* Pandas
* Numpy
* Plotly
* Seabon
* Matplotlib
* Gradio
 
## Credits 
* [The Code Institute](https://codeinstitute.net/) Learning Management System
* [VS Code](https://code.visualstudio.com/) was used to wite the code
* [ChatGPT](https://chatgpt.com/) was used to generate and debug code
* [README](https://github.com/Code-Institute-Solutions/da-README-template) template
* [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) data set was used for this project
* [gradio](https://www.gradio.app/)
