# Phonemes Classification

## DataSet 

The data were extracted from the TIMIT database (TIMIT Acoustic-Phonetic Conti-
nuous Speech Corpus, NTIS, US Dept of Commerce) which is a widely used resource for
research in speech recognition. A dataset was formed by selecting five phonemes for clas-
sification based on digitized speech from this database. The phonemes are transcribed as
follows : “sh” as in “she”, “dcl” as in “dark”, “iy” as the vowel in “she”, “aa” as the vowel in
“dark”, and “ao” as the first vowel in “water”. From continuous speech of 50 male speakers,
4509 speech frames of 32 msec duration were selected, approximately 2 examples of each
phoneme from each speaker. Each speech frame is represented by 512 samples at a 16kHz
sampling rate, and each frame represents one of the above five phonemes. The breakdown
of the 4509 speech frames into phoneme frequencies is as follows :

aa 695,
ao 1022,
dcl 757, 
iy 1163, 
sh 872.

From each speech frame, a log-periodogram was computed, which is one of several
widely used methods for casting speech data in a form suitable for speech recognition.
Thus the data used in what follows consist of 4509 log-periodograms of length 256,
with known class (phoneme) memberships. The learning dataset contains 2250 randomly
selected observations. It contains 256 columns labelled X1-X256 and a response column
Y.



