# Application of Artificial Neural Networks (ANN) to Automatic Speech Recognition (ASR) on a Novel Dataset created using YouTube
## Repository Structure Description
### Code
Contains 3 files. main.ipynb is a jupyter notebook used for partitioning an audio.wav,script.txt file pair into metadata.csv and phrases/*.wav to get ready for training and testing a model. main.py contains the main functionality for computing spectrograms for each audio sample, partitioning training and testing data, training the model using the CTC loss function, and validating the model against data it has never seen before. requirements.txt contains project dependencies and can be installed using the instructions in the usage section.
### Data
Data is a subdirectory containing the data used to train an test the network. Here we provide a small example data directory, but we describe in the following section how to expand the data directory on your own. Data contains a subdirectory phrases that is the result of main.ipynb. audio.wav is the audio file extracted from the YouTube video you wish to use. metadata.csv are id,phrase pairs for each phrase in the phrases subdirectory.
### Report
Contains the pdf of version 1 of the report.
### Presentation
Contains the pdf of version 1 and version 2 of the presentation.
### README.md
README.md is this file with information and instructions about the repository
## Usage
### Install Dependencies
*From the top level of this repository*
1. pip3 install Code/requirements.txt -r
### Download Dataset
*We provide an example dataset from a numberphile YouTube video that is small; however we describe how to use YouTube videos of your own choosing.*
1. Download the YouTube video of your choosing using the "Download" button below the YouTube video (requires YouTube subscription).
2. Extract the audio from the YouTube video in wav format using the popular ffmpeg command-line tool.
3. Name the audio "audio.wav".
4. Download the companion script from the YouTube video by clicking the 3 dots under the video player, highlighting the entire transcript, and pasting into a file named "script.txt".
5. Put "audio.wav" and "script.txt" in the "Data" subdirectory.
7. Run both cells in order.
8. You should have "metadata.csv" and "phrases/*.wav" in your "Data" subdirectory.
9. You are ready to train and evaluate a model.
### Train and Evaluate a Model
*You might have to adjust batch sizes according to the memory capacity of your machine.*
1. Adjust hyperparameters to your liking at the top of main.py.
2. Run main.py.