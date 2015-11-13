import sqlite3
from itertools import combinations
import pickle
import sys


def get_generator(statement):
    with sqlite3.connect("fanfiction_no_reviews.db") as conn:
        genc = conn.cursor()
        genc.execute(statement)
        ret = genc.fetchone()
        while ret is not None:
            yield ret
            ret = genc.fetchone()
        
def main(id):
    
    with open("authors.pkl", "r") as fp:
        authors = pickle.load(fp)
        authors = authors[id]
        
    with sqlite3.connect("fanfiction_links_%d.db" % id) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE links (node int, outlink int, PRIMARY KEY (node, outlink))")
    
    def dump(items):
        with sqlite3.connect("fanfiction_links_%d.db" % id) as conn:
            c = conn.cursor()
            for a, b in items:
                try:
                    c.execute("INSERT INTO links VALUES (?, ?)", (a, b))
                    c.execute("INSERT INTO links VALUES (?, ?)", (b, a))
                except Exception as e:
                    q = 0

    done = set()
    for author in authors:
        favs = set([x[0] for x in get_generator("SELECT storyID FROM author_favorites WHERE authorID = %d" % author)])
        #if len(favs) < 2: continue
        intersect = favs & done
        if len(intersect) < 2:
            dump([x for x in combinations(favs, 2)])
            done = done | favs
        else:
            newfavs = favs - intersect
            dump([x for x in combinations(newfavs, 2)])
            for cur in intersect:
                dump([(cur, x) for x in newfavs])
            done = done | newfavs
            
if __name__ == "__main__":
    main(int(sys.argv[1])-1)
