import sqlite3
from itertools import combinations
import time


def get_generator(statement):
    with sqlite3.connect("fanfiction_no_reviews.db") as conn:
        genc = conn.cursor()
        genc.execute(statement)
        ret = genc.fetchone()
        while ret is not None:
            yield ret
            ret = genc.fetchone()
        
class SimpleProgress:
    def __init__(self, total):
        self.total = total
    
    def start_progress(self):
        self.start_time = time.time()
        
    def update(self, x):
        if x>0:
            elapsed = time.time()-self.start_time
            percDone = x*100.0/self.total
            estimatedTimeInSec=(elapsed*1.0/x)*self.total
            return "%s %s percent\n%s Processed\nElapsed time: %s\nEstimated time: %s\n--------" % (self.bar(percDone), round(percDone, 2), x, self.form(elapsed), self.form(estimatedTimeInSec))
        return ""
    
    def expiring(self):
        elapsed = time.time()-self.start_time
        return elapsed/(60.0**2) > 71.
    
    def form(self, t):
        hour = int(t/(60.0*60.0))
        minute = int(t/60.0 - hour*60)
        sec = int(t-minute*60-hour*3600)
        return "%s Hours, %s Minutes, %s Seconds" % (hour, minute, sec)
        
    def bar(self, perc):
        done = int(round(30*(perc/100.0)))
        left = 30-done
        return "[%s%s]" % ('|'*done, ':'*left)
    
        
if __name__ == "__main__":
    conn = sqlite3.connect("fanfiction_no_reviews.db")
    c = conn.cursor()
    print "Doing initilizations"
    """c.execute("CREATE TABLE links (node int, outnode int, PRIMARY KEY (node, outnode))")
    c.execute("CREATE TABLE outcounts (node int, count int)")
    c.execute("CREATE TABLE names (name int)")
    length = 0
    for x in get_generator("SELECT storyID, COUNT(storyID) FROM author_favorites GROUP BY storyID"):
        c.execute("INSERT INTO outcounts VALUES (?,?)", (x[0], 0))
        c.execute("INSERT INTO names VALUES (?)", (x[0],))
        length += 1"""
    length = 3677236
    
    #c.execute("SELECT storyID, COUNT(storyID) FROM author_favorites GROUP BY storyID"); vals = c.fetchall()
    #c.executemany("INSERT INTO outcounts VALUES (?,?)", vals)
    #c.execute("SELECT COUNT(*) FROM outcounts"); length = len(vals)
    
    #c.execute("CREATE TABLE names (name int)")
    #c.executemany("INSERT INTO names VALUES (?)", [(x[0],) for x in vals])
    num = 296097

    t = 0
    
    def dump(items):
        
        for a, b in items:
            try:
                c.execute("INSERT INTO links VALUES (?, ?)", (a, b))
                c.execute("INSERT INTO links VALUES (?, ?)", (b, a))
            except Exception as e:
                q = 0

    done = set()
    print "Starting dumps"
    for author in get_generator("SELECT DISTINCT authorID FROM author_favorites WHERE authorID IN (SELECT authorID FROM author_favorites GROUP BY authorID HAVING COUNT(*) > 1) ORDER BY COUNT(authorID) ASC"):

        if t == 0:
             print "Starting timer"
             timer = SimpleProgress(num)
             timer.start_progress()
               
        cur = author[0]
        favs = set([x[0] for x in get_generator("SELECT storyID FROM author_favorites WHERE authorID = %d" % cur)])
        #if len(favs) < 2: continue
        intersect = favs & done
        if len(intersect) < 2:
            dump([x for x in combinations(favs, 2)])
            done = done | favs
        else:
            print "Doing Finegaling"
            newfavs = favs - intersect
            dump([x for x in combinations(newfavs, 2)])
            for cur in intersect:
                dump([(cur, x) for x in newfavs])
            done = done | newfavs
        t += 1
        print timer.update(t)
    conn.commit()
            

