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
import sys
import nltk
import os
import operator

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
        if "P" in filename:
            totalNumWordsP += len(tokens)
        else:
            totalNumWordsN += len(tokens)
        for tok in tokens:
            if tok not in tokenCount:
                tokenCount[tok] = 1
            else:
                tokenCount[tok] += 1

        for x in range(1, maxNGram+1):
            grams = nltk.ngrams(tokens, x)
            for gram in grams:
                uniqueNGrams[x-1].add(gram)

# Only keep tokens that occur at least 25 times (Task 4)
for tc in tokenCount:
    if tokenCount[tc] >= minCount:
        popTokens[tc] = tokenCount[tc]
sortedTokens = sorted(popTokens.items(), key=operator.itemgetter(1), reverse=True)

for x in range(len(uniqueNGrams)):
    print(str(x+1)+"-grams: " + str(len(uniqueNGrams[x])))

print("Top 10 words:")
for x in range(10):
    print(sortedTokens[x])

print("Rare words:")
for tc in tokenCount:
    if tokenCount[tc] < 5:
        print(tc + ": " + str(tokenCount[tc]))


# Task 5
def probabilityOfWGivenReviews(word, reviews, totalNumWords):
    frequency = 0
    for r in reviews:
        reviewText = normalize(open("train/" + r, 'r').read())
        frequency += reviewText.count(word)
    prob = 0.0 + frequency / totalNumWords
    return prob


for token in sortedTokens:
    p = probabilityOfWGivenReviews(token[0], posReviews, totalNumWordsP)
    if p != 0:
        positiveProbabilities[token[0]] = p

for token in sortedTokens:
    p = probabilityOfWGivenReviews(token[0], negReviews, totalNumWordsN)
    if p != 0:
        negativeProbabilities[token[0]] = p

print("positive probs:")
print(positiveProbabilities)
print("negative probs:")
print(negativeProbabilities)
for filename in os.listdir("test/"):
    probs = [1.0, 1.0] # p,n
    with open("test/" + filename, 'r') as myfile:
        reviewText = normalize(myfile.read())
        tks = nltk.word_tokenize(reviewText)
        for t in tks:
            if t in positiveProbabilities:
                probs[0] *= positiveProbabilities[t]
            if t in negativeProbabilities:
                probs[1] *= negativeProbabilities[t]
    if probs[0] < probs[1]:
        dclass = "N"
    else:
        dclass = "P"

    print("For "+filename+": " + str(probs)+" -> " + dclass)


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