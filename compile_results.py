import csv
from os import listdir
from os.path import isfile, join
import numpy as np

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
					if rank<authors[authorID]:
						authors[authorID]=rank
				else:
					authors[authorID] = rank
	
	ranks = [authors[x] for x in authors]
	rrs = [1./rank for rank in ranks]
	
	mrr = np.mean(rrs)
	mrr_sd = np.std(rrs)
	
	return mrr,mrr_sd,ranks
	# mrr = compute_MRR(ranks)
	# print authors
	#for cur in authors:
	#	ranks.append(compute_MRR(authors[cur]))
	#return sum(ranks) / float(len(ranks)), ranks

MRR,mrr_sd, ranks = calculate("/export/apps/dev/fanfiction/results",0)
print MRR
print mrr_sd
print ranks