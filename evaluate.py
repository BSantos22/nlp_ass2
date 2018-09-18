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
    for unwanted in unwantedTokens:
        text = text.replace(unwanted, "")
    return text.lower()


ground_truth_file = sys.argv[1]
results_file = sys.argv[2]

unwantedTokens = [',','.',':',';']
minCount = 25

tokenCount = {}
popularTokens = {}

maxNGram = 3

totalNumWordsP = 0
totalNumWordsN = 0
totalNumWords = 0

uniqueNGrams = [[set(),set()], [set(),set()], [set(),set()]]    # array of n-grams for n = (1, 2, 3), split up to positive / negative
bigramCount = {}                                                # count number of bigrams, split up to positive / negative
totalNumBigramsP = 0
totalNumBigramsN = 0
totalNumBigrams = 0

positiveProbabilities = {}                                      # dict key = token, value = probability
negativeProbabilities = {}                                      # dict key = token, value = probability

positiveProbabilitiesG = {}                                     # dict key = bigram, value = probability
negativeProbabilitiesG = {}                                     # dict key = bigram, value = probability

# Tokenizing (Task 4), count unique n-grams for n=1,2,3, count tokens
for filename in os.listdir("train/"):                           # go through training set
    with open("train/" + filename, 'r') as myfile:
        review = normalize(myfile.read())                       # remove some unwanted letters & to lower case
        tokens = nltk.word_tokenize(review)
        for tok in tokens:                                      # count occurrence of all tokens for positive / negative reviews
            if tok not in tokenCount:
                tokenCount[tok] = [0, 0]
            if "P" in filename:
                tokenCount[tok][0] += 1
            else:
                tokenCount[tok][1] += 1

        for x in range(1, maxNGram+1):                          # count unique grams (1, 2, 3) and bigrams in positive / negative reviews
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


for tc in tokenCount:                                           # Only keep tokens that occur at least 25 times
    if tokenCount[tc][0]+tokenCount[tc][1] >= minCount:
        popularTokens[tc] = tokenCount[tc]
sortedTokens = sorted(popularTokens.items(), key=operator.itemgetter(1), reverse=True)

for gramsPN in bigramCount:                                     # count total number of bigrams for later calculations
    values = bigramCount[gramsPN]
    totalNumBigrams += values[0] + values[1]
    totalNumBigramsP += values[0]
    totalNumBigramsN += values[1]

for tok in popularTokens:                                       # count total number of tokens for later calculations
    totalNumWordsP += popularTokens[tok][0]
    totalNumWordsN += popularTokens[tok][1]

totalNumWords = totalNumWordsP + totalNumWordsN

print("Top 10 words:")
for x in range(10):
    print(sortedTokens[x])

'''                                                             # commented out because it is a very long list
print("Rare words:")
for tc in tokenCount:
    if tokenCount[tc] < 5:
        print(tc + ": " + str(tokenCount[tc]))
'''

k_smoothing_task7 = 0.01  # best for task 7
k_smoothing_task5 = 0.37  # best for task 5


# Task 5
# return the probability of a token given a positive or negative review
def probabilityOfWGivenReviews(word, reviewType):
    if reviewType == 'P':
        return (popularTokens[word][0] + k_smoothing_task5) / (totalNumWordsP + k_smoothing_task5 * totalNumWords)
    return (popularTokens[word][1] + k_smoothing_task5) / (totalNumWordsN + k_smoothing_task5 * totalNumWords)


p = []                                                          # store the probabilities for all tokens (positive / negative)
for token in sortedTokens:
    p = probabilityOfWGivenReviews(token[0], 'P')
    if p != 0:
        positiveProbabilities[token[0]] = p

    p = probabilityOfWGivenReviews(token[0], 'N')
    if p != 0:
        negativeProbabilities[token[0]] = p


# Task 7
# return the probability of a bigram given a positive or negative review
def probabilityOfWGivenReviewsG(g, reviewType):
    if reviewType == 'P':
        return (bigramCount[g][0] + k_smoothing_task7) / (totalNumBigramsP + k_smoothing_task7 * totalNumBigrams)
    return (bigramCount[g][1] + k_smoothing_task7) / (totalNumBigramsN + k_smoothing_task7 * totalNumBigrams)


p = []                                                          # store the probabilities for all bigrams (positive / negative)
for gr in bigramCount:
    pp = probabilityOfWGivenReviewsG(gr, 'P')
    if pp != 0:
        positiveProbabilitiesG[gr] = pp

    pn = probabilityOfWGivenReviewsG(gr, 'N')
    if pn != 0:
        negativeProbabilitiesG[gr] = pn


# classification of a review (tks being the token of the review) according to task 5 (using tokens)
# returns an array of probabilities (positive / negative)
def classifyTask5(tks):
    probs = [0.0, 0.0]  # p,n
    for to in tks:
        if to in positiveProbabilities:
            probs[0] += math.log(positiveProbabilities[to])
        if to in negativeProbabilities:
            probs[1] += math.log(negativeProbabilities[to])
    return probs


# classification of a review (tks being the token of the review) according to task 7 (using bigrams)
# returns an array of probabilities (positive / negative)
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
for filename in os.listdir("test/"):                        # predict for each test-review
    probs = [0.0, 0.0]  # p,n
    with open("test/" + filename, 'r') as myfile:
        reviewText = normalize(myfile.read())
        tks = nltk.word_tokenize(reviewText)

        #probs = classifyTask7(tks)
        probs = classifyTask5(tks)

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
summary = str(correct) + " out of " + str(len(ground_truth_map)) + " were correct!\naccuracy " + str(acc)+"\n"
print(summary)
