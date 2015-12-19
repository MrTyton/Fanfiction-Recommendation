import csv
from os import listdir
from os.path import isfile, join
import numpy as np
import itertools

def recall_at(ranks, at=1000):
	recall = float(len([rank for rank in ranks if rank<=at]))/float(len(ranks))
	return recall

def calculate(mypath, split):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	onlyfiles = ["%s/%s" % (mypath, x) for x in onlyfiles]
	authors = {}
	
	for cur in onlyfiles:
		with open(cur, "r") as fp:
			r = csv.reader(fp)
			# authorID, favStoryID, fold, indexOfFav, len(total=heldOut)
			for i,row in enumerate(r):
				if i == 0: continue
				#row = row.split(",")
				#print row
				if int(row[2]) != split: continue
				#print "Not Skipping"
				rank = int(row[3])
				authorID = row[0]
				if authorID in authors:
					# appending index of favorite from heldout
					authors[authorID].append(rank)
				else:
					authors[authorID] = [rank]
	
	ranks = [np.min(authors[x]) for x in authors]
	rrs = [1./rank for rank in ranks]
	mrr = np.mean(rrs)
	mrr_sd = np.std(rrs)
	
	allranks = []
	for x in authors:
		allranks.extend(authors[x])
	recalls = [recall_at(allranks,cutoff) for cutoff in range(0,100000,1000)]
	return mrr,mrr_sd,recalls,ranks
	# mrr = compute_MRR(ranks)
	# print authors
	#for cur in authors:
	#	ranks.append(compute_MRR(authors[cur]))
	#return sum(ranks) / float(len(ranks)), ranks

def print_results(exp,loc,fold=0):
	
	MRR,mrr_sd,recalls, ranks = calculate(loc,fold)
	
	print("{} {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(exp,MRR, mrr_sd,recalls[1],recalls[49],recalls[99]))

print("Exp.,  MRR,  SD,    R@1k,  R@50k, R@100k")
print_results("CF   ", "/export/apps/dev/fanfiction/results4/all_results/results_cf")
print_results("SVD  ","/export/apps/dev/fanfiction/results4/all_results/results_svd")
print_results("COS  ","/export/apps/dev/fanfiction/results4/all_results/results_cos")
print_results("LDA  ","/export/apps/dev/fanfiction/results4/all_results/results_lda_filtered")
