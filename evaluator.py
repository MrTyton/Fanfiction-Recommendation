import sqlite3
import csv
import sys
import pickle
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix


class Evaluator():
	def __init__(self, recommender, storyIDs = None, authorIDs = None, full=False, split=0):
		self.recommender = recommender
		self.fold = split
		if storyIDs is None:
			with sqlite3.connect("/users/jegang/fanfiction_no_reviews.db") as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT storyID FROM author_favorites") if not full else c.execute("SELECT id FROM stories")
				self.storyIDs = [x[0] for x in c.fetchall()]
		else:
			self.storyIDs = storyIDs
		if authorIDs is None:
			with sqlite3.connect("/users/jegang/fanfiction_no_reviews.db") as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT authorID FROM author_favorites")
				self.authorIDs = [x[0] for x in c.fetchall()]
		else:
			self.authorIDs = authorIDs
			
		with sqlite3.connect("/users/jegang/fanfiction_no_reviews.db") as conn:
			c = conn.cursor()
			c.execute("SELECT COUNT(DISTINCT id) FROM stories")
			self.num_all = c.fetchone()[0]
			
	def evaluate(self):
		results = {}
		for cur in self.authorIDs:
			#total = {}
			with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT input FROM author_insert_splits_%d WHERE authorID = %d" % (self.fold, cur)); favorites = tuple(sorted([x[0] for x in c.fetchall()]))

			#for i,stor in enumerate(self.storyIDs):
			toDo = self.recommender.populate(favorites)

			if toDo is not None:
				toDo = list(set(self.storyIDs) & toDo)
			else:
				toDo = self.storyIDs
			print "Parsed down %d stories into %d stories" % (len(self.storyIDs), len(toDo))
			#total = {}

			#for i, stor in enumerate(toDo):
			#	total[stor] = self.recommender.favorite_likelihood(stor, favorites)
			
			total = self.recommender.compute_all(favorites)
			if total is None:
				total = [[stor,self.recommender.favorite_likelihood(stor,favorites)] for stor in toDo]
			#results[cur] = total
			total = sorted(total, key = lambda x:x[1], reverse=True)
			total_no_ranks = [x[0] for x in total]
			with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT output FROM author_output_splits_%d WHERE authorID = %d" % (self.fold, cur)); heldout = sorted([x[0] for x in c.fetchall()])

			ranks = []
			
			#num_all = len(total_no_ranks)
			for x in heldout:
				try:
					ind = total_no_ranks.index(x) +1
				except Exception as e:
					ind = self.num_all
				ranks.append((cur, x, self.fold, ind, self.num_all))
		#for auth in results:
			with open("../results/%d_split_%d.csv" % (cur, self.fold), "w") as fp:
				w = csv.writer(fp)#, results[cur].keys())
				#w.writeheader()
				w.writerow(["Author", "Story", "Fold", "Rank", "Total number of Stories"])
				w.writerows(ranks)
				
				
class base_cf():
	def __init__(self,fold, penalty = 20):
		self.done_favorite_lists = {}
		self.fold = fold
		self.penalty = float(penalty)
	
	def populate(self,favorites):
		if favorites not in self.done_favorite_lists: 
			with sqlite3.connect("../author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT input FROM author_insert_splits_%d WHERE authorID IN (SELECT authorID FROM author_insert_splits_%d WHERE input IN %s) AND input NOT IN %s" % (self.fold, self.fold, str(favorites), str(favorites)))
				potentials = [x[0] for x in c.fetchall()]
				self.done_favorite_lists[favorites] = set(potentials)
		return self.done_favorite_lists[favorites]
	
	def compute_all(self, favorites):
		return None
	
	def favorite_likelihood(self,stor, favorites): #require a TUPLE of favorites; if sorted will be much faster
		if favorites in self.done_favorite_lists:
			potentials = self.done_favorite_lists[favorites]
		else:
			with sqlite3.connect("../author_splits_%d.db" % self.fold) as conn:
				c = conn.cursor()
				c.execute("SELECT DISTINCT input FROM author_insert_splits_%d WHERE authorID IN (SELECT authorID FROM author_insert_splits_%d WHERE input IN %s) AND input NOT IN %s" % (self.fold, self.fold, str(favorites), str(favorites)))
				potentials = [x[0] for x in c.fetchall()]
				self.done_favorite_lists[favorites] = set(potentials)
		if stor not in potentials: print "Skipping"; return 0
		else:
			with sqlite3.connect("../author_splits_%d.db" % self.fold) as conn:
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
		
		with sqlite3.connect("/users/jegang/author_splits_%d.db" % self.fold) as conn:
			c = conn.cursor()
			c.execute("SELECT DISTINCT input FROM author_insert_splits_%d" % self.fold); stories = [x[0] for x in c.fetchall()]
			c.execute("SELECT DISTINCT authorID FROM author_insert_splits_%d" % self.fold); authors = [x[0] for x in c.fetchall()]
			c.execute("SELECT * FROM author_insert_splits_%d" % self.fold); everything = c.fetchall()
		self.story_keys = {val:i for i, val in enumerate(stories)}
		self.reverse_story_keys = {v:k for k,v in self.story_keys.items()}
		self.author_keys = {val:i for i, val in enumerate(authors)}
		
		author_rows = [self.author_keys[x[0]] for x in everything]
		story_columns = [self.story_keys[x[1]] for x in everything]
		
		self.matrix = coo_matrix(([1] * len(author_rows), (author_rows, story_columns))).tocsc()
		
		self.done = {}
		
		self.svd = TruncatedSVD(n_components=100)
		self.svd.fit(self.matrix)
		
		
	def convert_to_vector(self, favorites):
		favorites = [x for x in favorites if x in self.story_keys]
		story_columns = [self.story_keys[x] for x in favorites]
		return coo_matrix(([1] * len(story_columns), ([0] * len(story_columns), story_columns)),shape=(1, self.matrix.shape[1])).tocsc()
	
	def populate(self, favorites):
		return None
	
	def compute_all(self, favorites):
		print "Computing all"
		try:
			return self.done[favorites][self.story_keys[stor]]
		except KeyError as e:
			print "Converting"
			favs = self.convert_to_vector(favorites)
			res = self.svd.transform(favs)
			res = self.svd.inverse_transform(res)
			self.done[favorites] = res[0]
			print "Returning"
			return [[self.reverse_story_keys[i],v] for i, v in enumerate(self.done[favorites])]
			#return self.done[favorites][self.story_keys[stor]]		
		
	
	def favorite_likelihood(self, stor, favorites):
		try:
			return self.done[favorites][self.story_keys[stor]]
		except KeyError as e:
			favs = self.convert_to_vector(favorites)
			res = self.svd.transform(favs)
			res = self.svd.inverse_transform(res)
			self.done[favorites] = res[0]
			return self.done[favorites][self.story_keys[stor]]
	
		
		
if __name__ == "__main__":
	value = int(sys.argv[1])-1
	fold = int(sys.argv[2])
	with open("../authors.pkl", "r") as fp:
		authors = pickle.load(fp)[value*1408:(value+1)*1408]
	#base = base_cf(fold)
	with open("/users/jegang/svd_0.pkl", "r") as fp:
		SVD = pickle.load(fp)
	eval = Evaluator(SVD, authorIDs=authors, split=fold)
	eval.evaluate()