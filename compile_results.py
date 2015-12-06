import csv
from os import listdir
from os.path import isfile, join

def compute_MRR(ranks):
	return 1./len(ranks) * sum([1./x for x in ranks])
	
def calculate(mypath, split):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	onlyfiles = ["%s/%s" % (mypath, x) for x in onlyfiles]
	authors = {}
	
	for cur in onlyfiles:
		with open(cur, "r") as fp:
			r = csv.reader(fp)
			for i,row in enumerate(r):
				if i == 0: continue
				#row = row.split(",")
				#print row
				if int(row[2]) != split: continue
				#print "Not Skipping"
				if row[0] in authors:
					authors[row[0]].append(int(row[3]))
				else:
					authors[row[0]] = [int(row[3])]
	ranks = []
	print authors
	for cur in authors:
		ranks.append(compute_MRR(authors[cur]))
	return sum(ranks) / float(len(ranks)), ranks