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

uniqueNGrams = [[set(),set()], [set(),set()], [set(),set()]]  # ngram
bigramCount = {}
totalNumBigramsP = 0
totalNumBigramsN = 0
totalNumBigrams = 0

positiveProbabilities = {}
negativeProbabilities = {}
positiveProbabilitiesG = {}
negativeProbabilitiesG = {}
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
                if x == 2:
                    if gram not in bigramCount:
                        bigramCount[gram] = [0, 0]

                if 'P' in filename:

                    if x == 2:
                        bigramCount[gram][0] += 1
                    uniqueNGrams[x - 1][0].add(gram)
                if 'N' in filename:
                    if x == 2:
                        bigramCount[gram][1] += 1
                    uniqueNGrams[x - 1][1].add(gram)

# Only keep tokens that occur at least 25 times (Task 4)
for tc in tokenCount:
    if tokenCount[tc][0]+tokenCount[tc][1] >= minCount:
        popTokens[tc] = tokenCount[tc]
sortedTokens = sorted(popTokens.items(), key=operator.itemgetter(1), reverse=True)

print(bigramCount)
for gramsPN in bigramCount:
    values = bigramCount[gramsPN]
    totalNumBigrams += values[0] + values[1]
    totalNumBigramsP += values[0]
    totalNumBigramsN += values[1]

print(str(totalNumBigrams) + " -> " + str(totalNumBigramsP) + " + " + str(totalNumBigramsN))
print(len(uniqueNGrams[1][0])+len(uniqueNGrams[1][1]))

for tok in popTokens:
    totalNumWordsP += popTokens[tok][0]
    totalNumWordsN += popTokens[tok][1]

totalNumWords = totalNumWordsP + totalNumWordsN

for x in range(len(uniqueNGrams)):
    print(str(x+1)+"-grams: " + str(len(uniqueNGrams[x][0])+len(uniqueNGrams[x][1])))
print(sortedTokens)
print("Top 10 words:")
for x in range(10):
    print(sortedTokens[x])
    
'''
print("Rare words:")
for tc in tokenCount:
    if tokenCount[tc] < 5:
        print(tc + ": " + str(tokenCount[tc]))
'''

k_smoothing = 0.01

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


# Task 7
def probabilityOfWGivenReviewsG(g, reviewType):
   # print("prob for " + str(g))
    if reviewType == 'P':
        return (bigramCount[g][0] + k_smoothing) / (totalNumBigramsP + k_smoothing * totalNumBigrams)
    return (bigramCount[g][1] + k_smoothing) / (totalNumBigramsN + k_smoothing * totalNumBigrams)


p = []
for gr in bigramCount:
    pp = probabilityOfWGivenReviewsG(gr, 'P')
    if pp != 0:
        positiveProbabilitiesG[gr] = pp

    pn = probabilityOfWGivenReviewsG(gr, 'N')
    if pn != 0:
        negativeProbabilitiesG[gr] = pn

print("total #words p: "+str(totalNumWordsP))
print("total #words n: "+str(totalNumWordsN))
print("positive probs:")
print(positiveProbabilities)
print("negative probs:")
print(negativeProbabilities)


def classifyTask5(tks):
    probs = [0.0, 0.0]  # p,n
    for to in tks:
        if to in positiveProbabilities:
            probs[0] += math.log(positiveProbabilities[to])
        if to in negativeProbabilities:
            probs[1] += math.log(negativeProbabilities[to])
    return probs


def classifyTask7(tks):
    probs = [0.0, 0.0]  # p,n
    grms = nltk.ngrams(tks, 2)
    if tks[0] in positiveProbabilities:
        probs[0] += math.log(positiveProbabilities[tks[0]])
    elif tks[0] in negativeProbabilities:
        probs[1] += math.log(negativeProbabilities[tks[0]])

    for g in grms:
        if g in bigramCount:
            if g in positiveProbabilitiesG:
                probs[0] += math.log(positiveProbabilitiesG[g])
            if g in negativeProbabilitiesG:
                probs[1] += math.log(negativeProbabilitiesG[g])

    return probs


preds = []
for filename in os.listdir("test/"):
    probs = [0.0, 0.0]  # p,n
    with open("test/" + filename, 'r') as myfile:
        reviewText = normalize(myfile.read())
        tks = nltk.word_tokenize(reviewText)

        probs = classifyTask7(tks)
        #probs = classifyTask5(tks)

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
acc = float(correct)/len(ground_truth_map)
summary = "For k = " + str(k_smoothing) + ": " + str(correct) + " out of " + str(len(ground_truth_map)) + " were correct!\naccuracy " + str(acc)+"\n"
print(summary)
