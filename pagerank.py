import sqlite3
from itertools import combinations
import pickle
import sys
import time
from random import shuffle
from os import listdir, remove
from os.path import isfile, join
import shutil
import csv
from copy import deepcopy
import time
import multiprocessing
from natsort import natsorted




def get_generator(statement, file=None):
    with sqlite3.connect("fanfiction_no_reviews.db" if file is None else file) as conn:
        genc = conn.cursor()
        genc.execute(statement)
        ret = genc.fetchone()
        while ret is not None:
            yield ret
            ret = genc.fetchone()

def combine(files):
    shuffle(files)
    timer = SimpleProgress(len(files))
    t = 0
    with sqlite3.connect("fanfiction_links.db") as conn:
        c = conn.cursor()
        timer.start_progress()
        for curr in files:
            print timer.update(t)
            print "Adding %s" % curr
            c.execute("ATTACH '%s' AS toMerge" % curr)
            c.execute("INSERT OR IGNORE INTO links SELECT * FROM toMerge.links")
            c.execute("DETACH toMerge")
            conn.commit()
            t += 1
                    


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
            items = list(set([(x,y) for x, y in items] + [(y, x) for x, y in items]))
            c.executemany("INSERT OR IGNORE INTO links VALUES (?, ?)", items)
            """for a, b in items:
                try:
                    c.execute("INSERT INTO links VALUES (?, ?)", (a, b))
                    c.execute("INSERT INTO links VALUES (?, ?)", (b, a))
                except Exception as e:
                    q = 0"""

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
            
def combine_cluster(id):
    
    with sqlite3.connect("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.db" % id) as conn:
    #with sqlite3.connect("../fanfiction_sorted_links_%d.db" % id) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE links (node int, outlink int, PRIMARY KEY (node, outlink))")
    
    minval = id * 232033
    maxval = ((id + 1) * 232033) -1
    def dump(items, timeout=300):
        failed = 0
        with sqlite3.connect("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.db" % id, timeout) as conn:
        #with sqlite3.connect("../fanfiction_sorted_links_%d.db" % id, timeout) as conn:
            c = conn.cursor()
            #for a, b in items:
                
                #try:
            c.execute("PRAGMA main.synchronous=OFF")
            c.execute("PRAGMA temp.synchronous=OFF")
            c.execute("PRAGMA temp_store=MEMORY")
            c.executemany("INSERT OR IGNORE INTO links VALUES (?, ?)", items)
                #except sqlite3.IntegrityError as e:
                 #   failed += 1
                #except Exception as e:
                #    q=0
                    #print "Database %d excepted on values %s" % (id, (a, b)), e
            #print "Added %d links, discarded %d" % (len(items) - failed, failed)
    mypath = "H:/Fanfiction Data/test/Clusterized"
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles = [x for x in onlyfiles if '.db' in x and x != 'fanfiction_no_reviews.db']
    onlyfiles = [mypath + "/" + x for x in onlyfiles]
    shuffle(onlyfiles)
    onlyfiles = natsorted(onlyfiles)
    #onlyfiles.reverse()
    print "Thread %d starting crawl through files" % id
    for curfile in onlyfiles:
        print "Crawling through %s on thread %d" % (curfile, id)
        #while True:
            #try:
        vals = [x for x in get_generator("SELECT * FROM links WHERE node BETWEEN %d AND %d" % (minval, maxval), curfile)]
                #break
            #except sqlite3.DatabaseError as e:
            #    print "WTF fix this"
            #    raw_input()
        dump(vals)
    
    print "Thread %d creating outcounts" % id
    conn = sqlite3.connect("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.db" % id)
    c = conn.cursor()
    #c.execute("CREATE TABLE outcounts (node int PRIMARY KEY, nodecount int)")
    #c.execute("INSERT INTO outcounts SELECT node, count(node) FROM links GROUP BY node")
    c.execute("SELECT node, COUNT(node), GROUP_CONCAT(outlink) FROM links GROUP BY node ORDER BY COUNT(node) ASC")
    data = c.fetchall()
    conn.close()
        
    print "Thread %d converting to csv" % id
    with open("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.csv" % id, "w") as fp:
    #with open("../fanfiction_sorted_links_%d.csv" % id, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(['storyID', 'count', 'outlinks'])
        writer.writerows(data)
    
    #shutil.move("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.db" % id, "C:/Users/Joshua/Documents/Cluster/sorted/fanfiction_sorted_links_%d.db" % id)
    shutil.move("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.csv" % id, "G:/Fanfiction Data/sorted/fanfiction_sorted_links_%d.csv" % id) 
    remove("C:/Users/Joshua/Documents/Cluster/fanfiction_sorted_links_%d.db" % id)
    #shutil.move("../fanfiction_sorted_links_%d.csv" % id, "../sorted/fanfiction_sorted_links_%d.csv" % id) 
    #remove("../fanfiction_sorted_links_%d.db" % id)
    print "Thread %d is done" % id
    
if __name__ == "__main__":
   #main(int(sys.argv[1])-1)
   start = int(sys.argv[1])
   end = int(sys.argv[2])
   timer = SimpleProgress(start-end)
   t = 0
   timer.start_progress()
   for i in range(start, end, 1):
       print timer.update(t)
       combine_cluster(i)
       t += 1
"""for i in range(0, 500, 100):
      threads = []
      for q in range(i, i+):
          w = multiprocessing.Process(target=combine_cluster, args=(q,))
          w.start()
          threads.append(w)
      for q in threads:
          q.join()"""
   #combine_cluster(int(sys.argv[1])-1)
