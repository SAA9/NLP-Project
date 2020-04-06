import argparse
import processor

def print_trace(file, ts):
	trace_string = ""
	for tweet_id, correct_language, likely_language, score in ts:
		score_antilog = 2**float(score)
		if correct_language == likely_language:
			label = 'correct'
		else:
			label = 'wrong'
		trace_string += ''.join([str(tweet_id), '  ', str(likely_language), '  ', str(score_antilog), '  ', str(correct_language), '  ', str(label), '\n'])

	with open(file, "w", encoding="utf-8") as file:
		file.write(trace_string)

def print_eval(file, trace_accuracy, trace_precision, trace_recall, trace_F1, trace_macroF1_weighedF1):
	trace_eval = ""
	trace_eval += ''.join([str(trace_accuracy), '\n', str(trace_precision), '\n', str(trace_recall), '\n', str(trace_F1), '\n', str(trace_macroF1_weighedF1)])
	with open(file, "w", encoding="utf-8") as file:
		file.write(trace_eval)


def main():

	# python main.py -V '1' -n '1' -d '1' -tr 'training-tweets.txt' -te 'test-tweets-given.txt'
	# python main.py -V '1' -n '1' -d '1' -tr 'training-tweets.txt' -te 'test-tweets-given.txt' -b 

	parser = argparse.ArgumentParser(description='Trains or tests n-gram models')
	
	parser.add_argument("-V", help="vocabulary")
	parser.add_argument("-n", help="ngram")
	parser.add_argument("-d", help="smoothing")
	parser.add_argument("-tr", help="training")
	parser.add_argument("-te", help="testing")
	parser.add_argument("-b", help="byom", action='store_true', default=None)

	#command line input values
	args = parser.parse_args()
	vocabulary_number = int(args.V)
	ngram_number = int(args.n)
	smoothing_value = float(args.d)
	filename_training = args.tr
	filename_testing = args.te

	if args.b is None:
		byom = False
	else:
		byom = True

	training_set, training_tweet_data = processor.read_tweets(filename_training, vocabulary_number, ngram_number, byom)
	test_set, test_tweet_data = processor.read_tweets(filename_testing, vocabulary_number, ngram_number, byom)
	training_dictionary, vocabulary_sizes = processor.build_training_dictionary(training_set, ngram_number)
	ngram_probabilities = processor.calculate_ngram_characters_probability(training_dictionary, vocabulary_sizes, smoothing_value)
	predicted_result = processor.get_prediction_for_test(test_tweet_data, training_dictionary, vocabulary_sizes, ngram_probabilities, smoothing_value, ngram_number)

	result_tuple = []
	for test_tuple in predicted_result:
		tweet_id = test_tuple[2]
		correct_language = test_tuple[1]
		likely_language = predicted_result[test_tuple][0][0]
		score = predicted_result[test_tuple][0][1] 
		result_tuple.append((tweet_id, correct_language, likely_language, score))

	metrics_dictionary = processor.build_metrics_dictionary(result_tuple)
	trace_precision, trace_recall, trace_F1, trace_macroF1_weighedF1 = processor.evaluate_metrics(metrics_dictionary)
	trace_accuracy = processor.evalutate_accuracy(result_tuple, test_set)

	if (byom == False): 
		file_name_trace =  ''.join(['trace_',str(vocabulary_number), '_', str(ngram_number), '_', str(smoothing_value), '.txt'])
		file_name_eval = ''.join(['eval_',str(vocabulary_number), '_', str(ngram_number), '_', str(smoothing_value), '.txt'])
	else:
		file_name_trace =  ''.join(['trace_BYOM_',str(vocabulary_number), '_', str(ngram_number), '_', str(smoothing_value), '.txt'])
		file_name_eval = ''.join(['eval_BYOM_',str(vocabulary_number), '_', str(ngram_number), '_', str(smoothing_value), '.txt'])

	print_trace(file_name_trace, result_tuple)
	print_eval(file_name_eval, trace_accuracy, trace_precision, trace_recall, trace_F1, trace_macroF1_weighedF1)


if __name__ == "__main__":
	main()
