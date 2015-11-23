# coding: utf-8
import sqlite3

conn = sqlite3.connect("fanfiction_no_reviews.db")
c = conn.cursor()
c.execute("SELECT DISTINCT storyID FROM author_favorites")
stories = c.fetchall()
story_dict = {x[0]:i for i, x in enumerate(stories)}
c.execute("SELECT authorID, GROUP_CONCAT(storyID) FROM author_favorites GROUP BY authorID")
authors = c.fetchall()
authors = {x[0]: [story_dict[int(w)] for w in x[1].split(",")] for x in authors}
from scipy.sparse import dok_matrix
matrix = dok_matrix((len(authors), len(stories)), dtype=bool)
for i, cur in enumerate(authors):
    for val in authors[cur]:
        matrix[i, val] = True

matrix = matrix.tocsc()
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, n_iter=10)
svd.fit(matrix)
svd
svd.components_
svd.explained_variance_
