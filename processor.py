import math
import sys
import re

languages = ['eu', 'ca', 'gl', 'es', 'en', 'pt']
repeated_language_list = []

def read_tweets(filename, vocabulary, ngram_number, byom):
    tweets = [] 
    tweet_data = []

    file = open(filename, 'r', encoding='utf-8')
    tweet_lines = file.readlines()
    file.close()

    for line in tweet_lines:
        if (line != '\n'):
            tweet_ID, tweet = line.split(None, 1)
            username, tweet = tweet.split(None, 1)
            language, tweet = tweet.split(None, 1)

            repeated_language_list.append(language.strip())

            if(byom):
                tweet_alpha = ''.join(filter(str.isalpha, tweet))
                tweet_26 = re.sub("[^A-Za-z]+", ' ', tweet)
                tweet_lowercase_26 = tweet_26.lower()

                if(vocabulary == 0):
                    tweet = tweet_lowercase_26
                elif(vocabulary == 1):
                    tweet = tweet_26
                elif(vocabulary == 2):
                    tweet = tweet_alpha
                else:
                    print('vocabulary input needs to be between 0 & 2')
                    sys.exit()

            else:
                tweet_alpha = ''.join(filter(str.isalpha, line))
                tweet_26 = re.sub("[^A-Za-z]+", ' ', line)
                tweet_lowercase_26 = tweet_26.lower()
                if(vocabulary == 0):
                    tweet = tweet_lowercase_26
                elif(vocabulary == 1):
                    tweet = tweet_26
                elif(vocabulary == 2):
                    tweet = tweet_alpha
                else:
                    print('vocabulary input needs to be between 0 & 2')
                    sys.exit()

            if(ngram_number == 1):
                stripped_tweet = tweet.strip().replace(' ', '')
                tweets.append((stripped_tweet, language)) 
                tweet_data.append((stripped_tweet, language, tweet_ID))
            else:
                stripped_tweet = tweet.strip()
                tweets.append((stripped_tweet, language))  
                tweet_data.append((stripped_tweet, language, tweet_ID))
        else:
            break
    
    return tweets, tweet_data

def build_training_dictionary(training_set, ngram_number):
    training_dictionary = {}
    vocabulary_sizes = {}

    if(ngram_number ==1):
        for tweet, language in training_set:
            if language not in training_dictionary:
                training_dictionary[language] = tweet
            else:
                training_dictionary[language] += tweet

    elif(ngram_number == 2):
        for tweet, language in training_set:
            if language not in training_dictionary:
                training_dictionary[language] = []
            for (a, b) in zip(tweet[:-1], tweet[1:]):
                if str(a) != ' ' and str(b)!= ' ':
                    training_dictionary[language].append(str(a+b))
        

    elif(ngram_number == 3):
        for tweet, language in training_set:
            if language not in training_dictionary:
                training_dictionary[language] = []
            for (a, b, c) in zip(tweet[:-1], tweet[1:], tweet[2:]):
                if str(a) != ' ' and str(b)!= ' ':
                    training_dictionary[language].append(str(a+b+c))

    else:
        print('ngram_number need to be between 1 & 3')
        sys.exit()
        
    for language in training_dictionary:
        vocabulary_sizes[language] = len(set(training_dictionary[language]))
        print("vocabulary size for " + language + " = " + str(vocabulary_sizes[language]))

    return training_dictionary, vocabulary_sizes

def calculate_ngram_characters_probability(training_dictionary, vocabulary_sizes, smoothing_value):
    delta = smoothing_value
    ngram_probabilities = {}
    for key in training_dictionary:
    
        train_tweet = training_dictionary[key]
        total_ngrams = len(training_dictionary[key]) 

        ngrams = []
        unique_ngrams = vocabulary_sizes[key]

        for ngram in train_tweet:
            if ngram not in ngrams:
                if key not in ngram_probabilities: 
                    ngram_probabilities[key] = {ngram: ((train_tweet.count(ngram)+delta)/(total_ngrams + unique_ngrams * delta))}
                else: 
                    ngram_probabilities[key][ngram] = ((train_tweet.count(ngram)+delta)/(total_ngrams + unique_ngrams * delta ))
                ngrams.append(ngram)

    return ngram_probabilities


def get_prediction_for_test(test_tweet_data, training_dictionary, vocabulary_sizes, ngram_probabilities, smoothing_value, ngram_number):
    delta = smoothing_value
    predicted_result = {}
    repeated_language_list_count = len(repeated_language_list)

    for test_tuple in test_tweet_data:
        (test_tweet, _, _) = (test_tuple)
        
        if (ngram_number ==1):
            new_test_tweet = test_tweet

        elif (ngram_number == 2):
            new_test_tweet = []
            for (a, b) in zip(test_tweet[:-1], test_tweet[1:]):
                if str(a) != ' ' and str(b)!= ' ':
                    new_test_tweet.append(str(a+b))

        elif (ngram_number == 3):
            new_test_tweet = []
            for (a, b, c) in zip(test_tweet[:-1], test_tweet[1:], test_tweet[2:]):
                if str(a) != ' ' and str(b)!= ' ' and str(c)!= ' ':
                    new_test_tweet.append(str(a+b+c))
        else:
            print('ngram_number need to be between 1 & 3')
            sys.exit()
            
        sum_probabilities = 0
        lang_probs = {}
        for language in languages:
            total_ngrams = len(training_dictionary[language])
            unique_ngrams = vocabulary_sizes[language]

            probs = ngram_probabilities[language] 
            for ngram in new_test_tweet:
                if ngram in probs:
                    sum_probabilities += math.log2(probs[ngram])
                else:
                    sum_probabilities += math.log2(delta /(total_ngrams + unique_ngrams * delta))
        
            sum_probabilities+= math.log2(repeated_language_list.count(language) / repeated_language_list_count) 
            lang_probs[language] = sum_probabilities
            sum_probabilities = 0
        predicted_result[test_tuple] = sorted(lang_probs.items(), key=lambda x:x[1], reverse=True)

    return predicted_result


def evalutate_accuracy(result_tuple, test_set):
    trace_accuracy = ""
    accuracy = 0
    test_tweet_count = len(test_set)

    for _, correct_language, likely_language, _ in result_tuple:
        if correct_language == likely_language: 
                accuracy += 1.0

    accuracy/=test_tweet_count
    trace_accuracy += ''.join([str(accuracy)])

    return trace_accuracy

def build_metrics_dictionary(result_tuple):
    metrics_dictionary = {}
        
    for language in languages:
        for _, correct_language, likely_language, _ in result_tuple:
            True_Positive = 0
            False_Positive = 0
            False_Negative = 0
            
            if (correct_language==language and likely_language==language):
                True_Positive = 1

            elif (correct_language==language and likely_language!=language):	
                False_Negative = 1

            elif (correct_language!=language and likely_language==language):
                False_Positive = 1
                
            if (language in metrics_dictionary):
                metrics_dictionary[language] = (metrics_dictionary[language][0] + True_Positive, metrics_dictionary[language][1] + False_Positive, metrics_dictionary[language][2] + False_Negative)
            else:
                metrics_dictionary[language] = (True_Positive, False_Positive, False_Negative)
    
    return metrics_dictionary

def evaluate_metrics(metrics_dictionary):

    repeated_language_list_count = len(repeated_language_list)
    language_count = len(languages)
    
    trace_precision = ""
    trace_recall = ""
    trace_F1 = ""
    trace_macroF1_weighedF1 = ""
    F1_measure = 0
    F1_macro = 0
    F1_weighed = 0

    for language in languages:
    
        precision = 0 
        recall = 0

        True_Positive, False_Positive, False_Negative = metrics_dictionary[language]
        
        if (True_Positive + False_Positive) !=0:
            precision = 1.0 * True_Positive / (True_Positive + False_Positive) 
        if (True_Positive + False_Negative) !=0:
            recall = 1.0 * True_Positive / (True_Positive + False_Negative) 
        
        if not (precision==0 and recall==0):
            F1_measure = (2.0 * precision * recall) / (precision + recall)
            F1_macro += F1_measure
            F1_weighed += (repeated_language_list.count(language) * 2.0 * precision * recall) / (precision + recall)
        
        trace_precision += ''.join([str(precision), '  '])
        trace_recall += ''.join([str(recall), '  '])
        trace_F1 += ''.join([str(F1_measure), '  '])
        
    F1_macro /= language_count
    F1_weighed /= repeated_language_list_count

    trace_macroF1_weighedF1 += ''.join([str(F1_macro), '  ', str(F1_weighed)])

    return trace_precision, trace_recall, trace_F1, trace_macroF1_weighedF1