'''
R1
The first review (snippet) is positive, however it focuses on the actors only.
Issues for sentiment identification might be:
- the fact that a different, less well received, movie is mentioned
- the word disaster comes up, is however not related to the movie
R2
Clearly a negative review, should be easier to analyze than the previous one
due to obviously negative words
- some positive words, an identification software would need to figure out that
they're less significant
'''

from __future__ import print_function
from operator import itemgetter
import sys
import nltk
import os
import operator
import math

def read_predictions(filename):
    """ Read predictions into dictionary"""
    d = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            d[key] = val
    return d


def normalize(text):
    return text.lower().replace(unwanted, "")

k_smoothing = 1

# Read in arguments
ground_truth_file = sys.argv[1]
results_file = sys.argv[2]

unwantedTokens = [',','.',':',';']
minCount = 25

tokenCount = {}
popTokens = {}

maxNGram = 3

totalNumWordsP = 0
totalNumWordsN = 0
totalNumWords = 0

uniqueNGrams = [set(), set(), set()]  # ngram


positiveProbabilities = {}
negativeProbabilities = {}
posReviews = [fi for fi in os.listdir("train/") if "P" in fi]
negReviews = [fi for fi in os.listdir("train/") if "N" in fi]

# Tokenizing (Task 4), count unique n-grams for n=1,2,3, count tokens
for filename in os.listdir("train/"):
    with open("train/" + filename, 'r') as myfile:
        review = myfile.read()
        for unwanted in unwantedTokens:
            review = normalize(review)
        tokens = nltk.word_tokenize(review)
        for tok in tokens:
            if tok not in tokenCount:
                tokenCount[tok] = [0, 0]
            if "P" in filename:
                tokenCount[tok][0] += 1
            else:
                tokenCount[tok][1] += 1

        for x in range(1, maxNGram+1):
            grams = nltk.ngrams(tokens, x)
            for gram in grams:
                uniqueNGrams[x-1].add(gram)
# Only keep tokens that occur at least 25 times (Task 4)
for tc in tokenCount:
    if tokenCount[tc][0]+tokenCount[tc][1] >= minCount:
        popTokens[tc] = tokenCount[tc]
sortedTokens = sorted(popTokens.items(), key=operator.itemgetter(1), reverse=True)


for tok in popTokens:
    totalNumWordsP += popTokens[tok][0]
    totalNumWordsN += popTokens[tok][1]

totalNumWords = totalNumWordsP + totalNumWordsN

for x in range(len(uniqueNGrams)):
    print(str(x+1)+"-grams: " + str(len(uniqueNGrams[x])))

print("Top 10 words:")
for x in range(10):
    print(sortedTokens[x])
'''
print("Rare words:")
for tc in tokenCount:
    if tokenCount[tc] < 5:
        print(tc + ": " + str(tokenCount[tc]))
'''


# Task 5
def probabilityOfWGivenReviews(word, reviewType):
    if reviewType == 'P':
        return (popTokens[word][0] + k_smoothing) / (totalNumWordsP + k_smoothing * totalNumWords)
    return (popTokens[word][1] + k_smoothing) / (totalNumWordsN + k_smoothing * totalNumWords)


p = []
for token in sortedTokens:
    p = probabilityOfWGivenReviews(token[0], 'P')
    if p != 0:
        positiveProbabilities[token[0]] = p

    p = probabilityOfWGivenReviews(token[0], 'N')
    if p != 0:
        negativeProbabilities[token[0]] = p

print("total #words p: "+str(totalNumWordsP))
print("total #words n: "+str(totalNumWordsN))
print("positive probs:")
print(positiveProbabilities)
print("negative probs:")
print(negativeProbabilities)

preds = []

for filename in os.listdir("test/"):
    probs = [0.0, 0.0]  # p,n
    with open("test/" + filename, 'r') as myfile:
        reviewText = normalize(myfile.read())
        tks = nltk.word_tokenize(reviewText)
        for t in tks:
            if t in positiveProbabilities:
                probs[0] += math.log(positiveProbabilities[t])
            if t in negativeProbabilities:
                probs[1] += math.log(negativeProbabilities[t])
    if probs[0] < probs[1]:
        dclass = "N"
    else:
        dclass = "P"
    preds.append(filename.replace(".txt","")+"\t"+dclass+"\n")
    print("For "+filename+": " + str(probs)+" -> " + dclass)

open(results_file, 'w').writelines(preds)

results_map = read_predictions(results_file)
ground_truth_map = read_predictions(ground_truth_file)

# Calculate accuracy and print incorrect predictions
correct = 0
for ID in ground_truth_map:
    label = ground_truth_map[ID]
    if ID not in results_map:
        print("Missing predictions for " + ID)
    elif results_map[ID] == label:
        correct = correct + 1
    else:
        print("Incorrect " + ID)

# Print summary
print(str(correct) + " out of " + str(len(ground_truth_map)) + " were correct!")
print("accuracy " + str(float(correct)/len(ground_truth_map)))

print("")
print("")
print("=======================================================================")
print("Task 6")
print("")
print("")

step = 25
for i in range (0, 5):
    print('\nThreshold: ' + str(minCount))
    characteristic_words = []
    for word in popTokens:
        if (popTokens[word][0] + popTokens[word][1] >= minCount):
            p = probabilityOfWGivenReviews(word, 'P')
            n = probabilityOfWGivenReviews(word, 'n')
            characteristic_words.append((word, p/n))
    positive = sorted(characteristic_words, key=itemgetter(1), reverse=True)[:10]
    negative = sorted(characteristic_words, key=itemgetter(1))[:10]
    print ('- Characteristic words of positive reviews')
    pstr = ''
    for word in positive:
        print('\t' + word[0] + ' - ' + str(round(word[1], 3)))
    print ('- Characteristic words of negative reviews')
    nstr = ''
    for word in negative:
        print('\t' + word[0] + ' - ' + str(round(1/word[1], 3)))
    print(pstr)
    print(nstr)
    minCount = minCount + step

'''
The experiment was repeat 5 times, starting with a threshold of 25, and increasing it by 25 each iteration.

The results were as follows:
25
Positive - (bruno,muppet,kurtz,kermit,willard,granger,muppets,brynner,mario,anthony)
Negative - (aztec,robot,heist,awful,julie,mummy,ocean,armored,waste,budget)
50
Positive - (bruno,muppet,kurtz,kermit,walker,hitchcock,excellent,game,murder,train)
Negative - (mummy,house,worst,money,boring,bad,minutes,nothing,no,trying)
75
Positive - (bruno,hitchcock,mother,war,wife,performance,best,great,show,guy)
Negative - (bad,minutes,no,looking,plot,even,?,then,thought,do)
100
Positive - (bruno,hitchcock,war,best,great,show,guy,still,most,us)
Negative - (bad,no,plot,even,?,then,do,horror,watching,why)
125
Positive - (war,best,great,guy,most,also,love,now,many,man)
Negative - (bad,no,plot,even,?,then,do,horror,only,n't)

As it was to be expected, when the threshold is too small a lot of rare words appear at the top, which don't really give any insight on what's positive or negative.
When the threshold increases, we begin to see the emergence of adjectives such as 'excellent', 'best' and 'great' for the positive reviews and 'worst', 'boring' and 'bad' for the negative ones.
It's also interesting to note that even at higher thresholds words such as 'hitchcock' and 'war' continue to appear highly rated on positive reviews, and 'horror' appears on negative reviews.
This could either mean that reviewers prefer these kind of movies, or point at the general quality of genres/director.
'''