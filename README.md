# Spotify Recommendation
### Created by
- Kevin Alvarez
- Will Burns
- Jenna Jabourian
- Connor Perrone
- Gabe Sanchez

---

### Dataset
We used the [spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) which contains information on 114,000 tracks on Spotify spanning over 125 genres.

The data is in CSV format, with a single track being the observational unit in each row. The Hugging Face Dataset card defines the columns as:

| Column (Feature)  | Description               |
| ----------------  | ------------------------- |
| **track_id**      | The track's Spotify ID.   |
| **artists**       | The track's artist(s). Separated by `;` if there are more than one. |
| **album_name**    | The track's album name.   |
| **track_name**    | The track's name.         |
| **popularity**    | An integer between 0 and 100 indicating how popular a song is with 100 being the most. This value is calculated by an algorithm that is based on the number of plays this track has and how recent those plays were (more plays recently means higher populatity). |
| **duration_ms**   | The track's length in milliseconds. |
| **explicit**      | 1 if the track contains explicit lyrics, or 0 if it is either unknown or does not have explicit lyrics. |
| **danceability**  | A value between 0.0 and 1.0 indicating how well a song could be danced to based on tempo, rhythm stability, beat strength, and overall regularity with 1.0 being the most danceable. |
| **energy**        | A value between 0.0 and 1.0 indicating how energetic the track is perceived based on how fast, loud and noisy it is with 1.0 being the most energetic. For example, metal tracks are high energy and classical tracks are low energy. |
| **key**           | The track's key as an integer in standard Pitch Class notation (e.g. C = 0, C#/Db = 1, D = 2, etc.) or -1 if no key was detected. |
| **loudness**      | How loud the track is in decibels (dB). |
| **mode**          | The track's modality with major represented as 1 and minor represented as 0. |
| **speechiness**   | A value between 0.0 and 1.0 indicating how speech-like the track is. For example, a talk show, audio book, or poetry would be closer to 1.0. Values above 0.66 indicate a track that is probably consisting of only spoken words. Values between 0.33 and 0.66 indicate a track that has both spoken words and music that can be layered or in sections (which could be something like rap music). Values below 0.33 are most likely music and tracks that are not speech-like. |
| **acousticness**  | A value between 0.0 and 1.0 indicating the confidence for if a track is acoustic or not (1.0 is likely acoustic). |
| **instrumentalness** | A value between 0.0 and 1.0 indicating how likely a track is to have no vocals. Rap and spoken word are both "vocal" but "ooh" and "aah" sounds are considered instrumental. A value closer to 1.0 indicates a track with likely no vocal content. |
| **liveness**      | A value between 0.0 and 1.0 indicating the presence of a live audience. Higher values represent a higher probability the track was performed live, and values above 0.8 indicate a high likelihood the track is live. |
| **valence**       | A value between 0.0 and 1.0 indicating how positive the track feels with a high value representing a happy and cheerful sounding track and a low value representing a sad or angry sounding track. |
| **tempo**         | The track's tempo in beats per minute (BPM). |
| **time_signature** | The track's time signature (indicating the number of beats per bar) ranging from 3 to 7 (representing time signatures of 3/4 to 7/4). |
| **track_genre**   | The track's genre. |


---

### Problem
Everyone has different prefernces for the music they listen to, but there is often an underlying "feeling" associated with songs we like. To find new songs similar to those we enjoy, we could try to search by describing this "feeling". But that can only get us so far.

By using the Spotify dataset we built a recommendation system that mathematically finds this "feeling" so the user can find more songs they enjoy based on their provided reference songs.

---

### Methodology
We approached this problem by trying to implement several different algorithms. First, we tried to implement KNN to develop an initial recommendation system. This was based on the idea that we could find similar songs using distances from one point to nearest featuers as described above. Then, this was combined with the Random Forest model for popularity prediction. The underlying reasoning behind this was to try to recommend songs that were both predicted to be popular, but also similar to the initial song. To better capture nonlinear relationships between the many audio features, our final implementation of this recommendation system utilizes a neural network autoencoder. This ultimately helped build our recommendation system because the autoencoder learns a compressed embedding of each track by training the network to reconstruct the original input from a reduced latent space. This is particularly helpful for our recommendations system as our dataset looks at many different audio characteristics, meaning it is helpful to look across multiple dimensions to find songs with similar features. We used ReLU as our activation function and MSE as our loss function. To evaluate model performance, we measured reconstruction loss across epochs and found values for the MSE, MAE, RMSE, and R^2 performance metrics.

---

### Results
TODO

---

### Usage
How to run the code (TODO)

For reference, this link shows how to download a notebook as a PDF

https://stackoverflow.com/questions/52588552/google-co-laboratory-notebook-pdf-download
