import sqlite3
from sklearn.feature_extraction import DictVectorizer



def base_cf(curUser): #grabs data from fanfiction_no_reviews; should change if working with another dataset
	conn = sqlite3.connect("fanfiction_no_reviews.db")
	c = conn.cursor()

	c.execute("SELECT DISTINCT storyID FROM author_favorites WHERE authorID IN (SELECT authorID FROM author_favorites WHERE storyID IN (SELECT storyID FROM author_favorites WHERE authorID = ?)) AND storyID NOT IN (SELECT storyID FROM author_favorites WHERE authorID = ?)", (curUser, curUser)); stories = c.fetchall(); stories = [x[0] for x in stories]

	#c.execute("SELECT DISTINCT storyID FROM author_favorites"); stories = c.fetchall(); stories = [x[0] for x in stories]

	rankings = []

	for i,curStory in enumerate(stories):
		c.execute("SELECT DISTINCT authorID FROM author_favorites WHERE storyID = ?", (curStory,)); authors = c.fetchall(); authors = [x[0] for x in authors if x[0] != curUser]

		ranking = 0.
		for cur in authors:
			c.execute("SELECT COUNT(*) FROM author_favorites w INNER JOIN (SELECT storyID FROM author_favorites WHERE authorID = ?) as q ON q.storyID = w.storyID WHERE w.authorID = ?", (cur, curUser))
			num = float(c.fetchone()[0])
			if num != 0:
				c.execute("SELECT num FROM author_counts WHERE id = ?", (cur,))
				dem = float(c.fetchone()[0] + 20.)
			else:
				dem = 1
			ranking += (num / dem)**2
		rankings.append((curStory, ranking))
		
	rankings = sorted(rankings, key=lambda x: x[1], reverse=True)
	conn.close()
	return rankings

def make_story_vectors():
	with sqlite3.connect("fanfiction_no_reviews.db") as conn:
		c = conn.cursor()
		c.execute("SELECT * FROM stories"); stories = c.fetchall()
		c.execute("SELECT storyid, GROUP_CONCAT(tag, '--------') FROM story_tags GROUP BY storyid"); tags = c.fetchall(); tags = {x[0]:x[1].split("--------") for x in tags}
	total = {}
	for x in stories:
		q = {}
		for w, e in zip(['author', 'title', 'words', 'published', 'updated', 'reviews', 'chapters', 'completed', 'category', 'rating', 'language'], x[1:-1]): q[w]=e
		del q['title']
		
		if x[0] in tags: 
			for temp in tags[x[0]]: q[temp] = True
		total[x[0]] = q
		
	return total
		

def train_feature_matrix(story_vectors = None):
	if story_vectors is None:
		total = make_story_vectors()
	else:
		total = story_vectors
	total = total.values()
	vec = DictVectorizer()
	vec.fit(total)
	return vec
	
# can probably use this matrix to compute the cosine vector for a user, then do clustering on that.

vec = train_feature_matrix()

#get stories here

#run the creation of the feature vectors here

author_vector = sum([vec.transform(x) for x in story_vectors]) / len(story_vectors)

#create evaluation set for only people with more than 10 favorites