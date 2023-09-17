# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import numpy as np

'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace= 1, bigram_laplace=0.005, bigram_lambda=0.6, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    uni_positive_b = Counter()
    uni_negetive_b = Counter()
    bi_positive_b = Counter()
    bi_negetive_b = Counter()
    bi_neg_c = 0
    bi_pos_c =0
    pos_c =0
    neg_c =0
    for review, label in zip(train_set, train_labels):
        if (label == 1):
            for words in review:
                words = words.lower()
                uni_positive_b.update([words])
                pos_c += 1

        if (label == 0):
            for words in review:
                words = words.lower()
                uni_negetive_b.update([words])
                neg_c += 1

    for review, label in zip(train_set, train_labels):
        if (label == 1):
            for i in range(len(review)-1):
                words = review[i].lower() + ' ' + review[i+1].lower()
                bi_positive_b.update([words])
                bi_pos_c += 1

        if (label == 0):
             for i in range(len(review)-1):
                words = review[i].lower() + ' ' + review[i+1].lower()
                bi_negetive_b.update([words])
                bi_neg_c += 1
    

            
    yhats =[]
    unique_c = len(uni_negetive_b) + len(uni_positive_b)
    bi_unique_c = len(bi_negetive_b) +len(bi_positive_b)
    for doc in tqdm(dev_set, disable=silently):
        pos_prob =0
        neg_prob =0
        bi_pos_prob =0
        bi_neg_prob = 0
        mix_neg = 0
        mix_pos = 0
        for word in doc:
            l_word = word.lower()
            pos_prob += math.log((uni_positive_b[l_word] +  unigram_laplace)/(pos_c + unigram_laplace*(unique_c +1)))
            neg_prob += math.log((uni_negetive_b[l_word] +  unigram_laplace)/(neg_c + unigram_laplace*(unique_c +1)))

        for i in range(len(doc)-1):
            words = doc[i] + ' ' + doc[i+1]
            l_word = words.lower()
            bi_pos_prob += math.log((bi_positive_b[l_word] +  bigram_laplace)/(bi_pos_c + bigram_laplace*(bi_unique_c +1)))
            bi_neg_prob += math.log((bi_negetive_b[l_word] +  bigram_laplace)/(bi_neg_c + bigram_laplace*(bi_unique_c +1)))


        neg_prob = neg_prob + math.log(1 - pos_prior)
        pos_prob = pos_prob + math.log(pos_prior)
        bi_neg_prob = bi_neg_prob + math.log(1 - pos_prior)
        bi_pos_prob = bi_pos_prob + math.log(pos_prior)

        mix_neg =( (1-bigram_lambda) * neg_prob)+ (bigram_lambda* bi_neg_prob)
        mix_pos =( (1-bigram_lambda) * pos_prob )+ (bigram_lambda* bi_pos_prob)

        if mix_pos >= mix_neg:
            a = 1
        else:
            a = 0

        yhats.append(a)

    
    return yhats
