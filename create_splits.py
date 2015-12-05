import sqlite3
from random import shuffle
from math import ceil

def create_splits(k=5, min=10, authors=None):
	with sqlite3.connect("fanfiction_no_reviews.db") as conn:
		c = conn.cursor()
		print "Getting Authors"
		if authors is None:
			c.execute("SELECT authorID, GROUP_CONCAT(storyID) FROM author_favorites GROUP BY authorID HAVING count(authorID) >= ?", (min,))
			authors = c.fetchall()
			authors = {x[0]:[int(y) for y in x[1].split(",")] for x in authors}
		
		for x in authors: shuffle(authors[x])
		print "Creating Splits"
		for i in range(k):
			print "Doing Split %d" % i
			input_insertions = []
			output_insertions = []
			for cur in authors:
				num = len(authors[cur])
				split = int(ceil(num/float(k)))
				output = authors[cur][i*split:(i+1)*split]
				input = authors[cur][:i*split] + authors[cur][(i+1)*split:]
				for x in input: input_insertions.append((cur, x))
				for x in output: output_insertions.append((cur, x))
			c.execute("CREATE TABLE author_insert_splits_%d (authorID int, input int)" % i)
			c.execute("CREATE TABLE author_output_splits_%d (authorID int, output int)" % i)
			c.executemany("INSERT INTO author_insert_splits_%d VALUES (?,?)" % i, input_insertions)
			c.executemany("INSERT INTO author_output_splits_%d VALUES (?,?)" % i, output_insertions)
			c.execute("CREATE INDEX indx_input_split_%d_1 ON author_insert_splits_%d(authorID)" % (i, i))
			c.execute("CREATE INDEX indx_ouptut_split_%d_1 ON author_output_splits_%d(authorID)" % (i, i))
			c.execute("CREATE INDEX indx_input_split_%d_2 ON author_insert_splits_%d(input)" % (i, i))
			c.execute("CREATE INDEX indx_ouptut_split_%d_2 ON author_output_splits_%d(output)" % (i, i))
		return authors
		
		