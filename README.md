# Classification of musical instruments

**Authors:** *Yotam Dafna, Matti Stern*

## Project goal
The aim of our project is to classify musical instruments using different machine learning algorithms and to investigate the behavior of the various instruments and their effect on the results obtained.

## Data-set description
Our repository includes audio (wav) audio clips of various instruments, with the name of the file being the playing instrument.
We took two repositories from two different places where one (called set-data) includes music clips in which only a single instrument is played, the other (called TrainingData-IRMAS) includes music clips in which several
Different instruments where the name of the file represents these instruments in the order of their dominance (from left to right) and finally also the genre of music.
Beyond splitting the data into train and test, we brought another data-set (called TestingData-IRMAS) that contains songs clips, where each wav file has a corresponding txt file that contains the tools that play there. We used this repository to try to predict which instruments are playing in the song.
We uploaded the files through the s’parser we wrote (depending on the database you want to upload).

## Algorithms - Sklearn
* KNN
* SVM
* Random Forest
* Adaboost

## The challenges we had and the solutions we found
* **Challenge 1: Dataset selection** <br>
  During the selection of the database we had a lot of thinking about which database we want.
  Should we choose a repository that contains audio clips of individual tools, or a combination of tools, are short audio clips of a few seconds enough or do we need longer audio clips, can choosing one of them make it difficult for us later when we want to perform more complex classifications and the like.
  <br>
  **Solution:** <br>
  Finally, we chose to use 3 different databases and make combinations in them to allow classification in different forms, and in particular to answer the questions we asked at the beginning.

* **Challenge 2: Convert audio to numeric vector.** <br>
  In order to study and classify the information using the various algorithms, we had to convert the audio files into a meaningful numerical vector. In order to do this we had to learn and understand how audio works, how it is represented according to different diagrams and what parameters exist in it.
  <br>
  **Solution:** <br>
  After reading and understanding how the audio files work, we went to a library called **librosa** that allows you to upload an audio file and extract information from it that can be used to assemble the vector.
  In the project we built 2 functions for creating vectors, one of which converts each audio file to a vector of 13 sizes.
  When it includes the information we saw as most relevant (best affects classification), and the second converts to a size 2 vector (contains the 13 and another 7 additional parameters of the information about the nature of the audio file).
  The reason for creating 2 different functions is that we wanted to test whether the success rates could be improved by adding more information to the vector.

* **Challenge 3: How to label the samples correctly.** <br>
  When we went to work on a repository of integrated musical instruments, we encountered classification problems (low success rates, and a large number of different classes).
  So we started trying to figure out what was the best way to tag Such a sample.
  (As you may recall, the file name of each sample is built in the order of dominance of the tools and finally the genre is specified)
  <br>
  **Solution:** <br>
  We tried to classify in a number of different methods, and compared the results we obtained.
  - Method 1: Labeling for a file will be according to the most dominant instrument (first from left only).
  - Method 2: The tagging for a file will be the names of all the tools, regardless of genre.
  - Method 3: Labeling for a file will be the most dominant tool and genre type
  - Method 4: Classification solely by genre.


* **Challenge 4: Classification of musical instruments in a single song as input.** <br>
  Our goal here was to try to identify the instrument in a single song that is received as input.
  Our input is an audio clip (wav), and a file (txt), which describes the instruments that are played in the song.
  The main problem was that when we try to predict the instrument in a song with the help of learning algorithms we get a single classification of instrument.
  The use of an integrated musical instrument database also did not help as not all combinations exist.
  <br>
  **Solution:** <br>
  So we thought of an idea and built an algorithm that works like this:
  1. Cut the song into chunks a second long.
  2. Perform a predict on the sections.
  3. Arrange the array according to the number of performances <br>
    (the most classified musical instrument will be in the first place and so on..)
  4. Perform unique on the array.
  5. Return the first k instruments when k = the number of instruments of the original song received as input. 
  
  

  