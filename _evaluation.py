import sys


def MRR(answers, rankings):
	rank = 0
	for cur in answers:
		index = rankings.find(answers)
		if index == -1:
			index = len(rankings)
		rank += 1. / index
	return rank
	
def evalatuate(favorites, answers, func):
	rankings = func(favorites)
	rankings = sorted(rankings)
	return MRR(answers, rankings)
	
	
def baseline
	