import sqlite3
import csv
import sys
import pickle
import logging

class Evaluator():
	def __init__(self, recommender, storyIDs = None, authorIDs = None, full=False, split=0, datadir="/users/jegang",resultsdir="../results"):
		self.recommender = recommender
		self.fold = split
		self.datadir = datadir
		self.resultsdir = resultsdir
		if storyIDs is None:
			with sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.datadir)) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT storyID FROM author_favorites") if not full else c.execute("SELECT id FROM stories")
				self.storyIDs = [x[0] for x in c.fetchall()]
		else:
			self.storyIDs = storyIDs
		if authorIDs is None:
			with sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.datadir)) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT authorID FROM author_favorites")
				self.authorIDs = [x[0] for x in c.fetchall()]
		else:
			self.authorIDs = authorIDs
			
	def evaluate(self):
		results = {}
		out = "{}/results.txt"
		rout=open(out, "w")
		for cur in self.authorIDs:
			#total = {}
			with sqlite3.connect("{}/author_splits_{}.db".format(self.datadir, self.fold)) as conn:
				c = conn.cursor()
				c.execute("SELECT input FROM author_insert_splits_%d WHERE authorID = %d" % (self.fold, cur)); favorites = tuple(sorted([x[0] for x in c.fetchall()]))

			#for i,stor in enumerate(self.storyIDs):
			if len(favorites)==0:
				logging.error("There are no favorites for author={}. Skipping.".format(cur))
				continue
			toDo = self.recommender.populate(favorites)
			if toDo is not None:
				toDo = list(set(self.storyIDs) & toDo)
			else:
				toDo = self.storyIDs
			print "Parsed down %d stories into %d stories" % (len(self.storyIDs), len(toDo))
			#total = {}

			#for i, stor in enumerate(toDo):
			#	total[stor] = self.recommender.favorite_likelihood(stor, favorites)
			
			total = {stor:self.recommender.favorite_likelihood(stor,favorites) for stor in toDo}
			results[cur] = total
			rout.write("{},{},{}\n".format(cur,self.fold,total))
		
		for auth in results:
			with open("{}/{}_split_{}.csv".format(self.resultsdir,auth, self.fold), "w") as fp:
				w = csv.DictWriter(fp, results[auth].keys())
				w.writeheader()
				w.writerow(results[auth])
				
				
class base_cf():
	def __init__(self,fold, penalty = 20):
		self.done_favorite_lists = {}
		self.fold = fold
		self.penalty = float(penalty)
	
	def populate(self,favorites):
		if favorites not in self.done_favorite_lists: 
			with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				# other stories from authors with these favorites
				c.execute("SELECT DISTINCT input FROM author_insert_splits_%d WHERE authorID IN (SELECT authorID FROM author_insert_splits_%d WHERE input IN %s) AND input NOT IN %s" % (self.fold, self.fold, str(favorites), str(favorites)))
				potentials = [x[0] for x in c.fetchall()]
				self.done_favorite_lists[favorites] = set(potentials)
		return self.done_favorite_lists[favorites]
	
	def favorite_likelihood(self,stor, favorites): #require a TUPLE of favorites; if sorted will be much faster
		if favorites in self.done_favorite_lists:
			potentials = self.done_favorite_lists[favorites]
		else:
			with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT input FROM author_insert_splits_%d WHERE authorID IN (SELECT authorID FROM author_insert_splits_%d WHERE input IN %s) AND input NOT IN %s" % (self.fold, self.fold, str(favorites), str(favorites)))
				potentials = [x[0] for x in c.fetchall()]
				self.done_favorite_lists[favorites] = set(potentials)
		if stor not in potentials: print "Skipping"; return 0
		else:
			with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT authorID FROM author_insert_splits_%d WHERE input = %d" % (self.fold, stor)); authors = [x[0] for x in c.fetchall()]
				ranking = 0
				for cur in authors:
					c.execute("SELECT COUNT(*) FROM author_insert_splits_%d WHERE authorID = %d AND input IN %s" % (self.fold, cur, str(favorites)))
					num = c.fetchone()[0]
					if num != 0:
						c.execute("SELECT num FROM author_counts WHERE authorID = %d" % (cur,))
						dem = c.fetchone()[0] + self.penalty
					else:
						dem = 1.
					ranking += (num / dem) ** 2
		return ranking
		
class svd():
	def __init__(self, fold):
		self.fold = fold
		
		
if __name__ == "__main__":
	value = int(sys.argv[1])
	fold = int(sys.argv[2])
	with open("../authors.pkl", "r") as fp:
		author = pickle.load(fp)[value]
	base = base_cf(fold)
	eval = Evaluator(base, authorIDs=[author], split=fold)
	eval.evaluate()