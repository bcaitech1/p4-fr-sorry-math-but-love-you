import editdistance
import numpy as np

def final_metric(sentence_acc: float, word_error_rate: float) -> float:
	return 0.9*sentence_acc + 0.1*(1-word_error_rate)


def word_error_rate(predicted_outputs, ground_truths):
	sum_wer=0.0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		output=output.split(" ")
		ground_truth=ground_truth.split(" ")
		distance = editdistance.eval(output, ground_truth)
		length = max(len(output),len(ground_truth))
		sum_wer+=(distance/length)
	return sum_wer / len(predicted_outputs)


def sentence_acc(predicted_outputs, ground_truths):
	correct_sentences=0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		if np.array_equal(output,ground_truth):
			correct_sentences+=1
	return correct_sentences / len(predicted_outputs)`