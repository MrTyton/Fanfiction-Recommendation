import sqlite3
from itertools import combinations
import time
import multiprocessing



def get_generator(statement):
    conn = sqlite3.connect("fanfiction_no_reviews.db")
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
    
    
    
    
def updater(consumequeue, stop, lock):
    print "In consumer thread"
    conn = sqlite3.connect("fanfiction_no_reviews.db")
    c = conn.cursor()
    while not stop.is_set():
        a, b = consumequeue.get()
        lock.acquire()
        try:
            c.execute("INSERT INTO links VALUES (?, ?)", (a, b))
            c.execute("INSERT INTO links VALUES (?, ?)", (b, a))
        except Exception:
            q = 0
        c.execute("UPDATE outcounts SET count = count + 1 WHERE node = ? OR node = ?", (a, b))
        lock.release()
        consumequeue.task_done()
      
def processer(jobqueue, consumequeue, timequeue, lock):
    print "In processing thread"
    conn = sqlite3.connect("fanfiction_no_reviews.db")
    c = conn.cursor()
    while not jobqueue.empty():
        cur = jobqueue.get()
        lock.acquire()
        favs = [x[0] for x in get_generator("SELECT DISTINCT storyID FROM author_favorites WHERE authorID = %d" % cur)]
        lock.release()
        for a,b in combinations(favs, 2):
            consumequeue.put((a, b))
        timequeue.put(1)
        jobqueue.task_done()
        
def timer(timequeue, stop, lock):
    print "In timer thread"
    conn = sqlite3.connect("fanfiction_no_reviews.db")
    c = conn.cursor()
    lock.acquire()
    c.execute("SELECT COUNT(DISTINCT authorID) FROM author_favorites"); num = c.fetchone()[0]
    lock.release()
    print "Starting timer"
    timer = SimpleProgress(num)
    timer.start_progress()
    t = 0
    while not stop.is_set():
        timequeue.get()
        t += 1
        print timer.update(t)
        timequeue.task_done()
        
        
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
    #c.execute("SELECT COUNT(DISTINCT authorID) FROM author_favorites"); num = c.fetchone()[0]
    #timer = SimpleProgress(num)
    #timer.start_progress()
    #t = 0
    print "Starting multiprocessing"
    jobqueue = multiprocessing.JoinableQueue()
    consumequeue = multiprocessing.JoinableQueue()
    timequeue = multiprocessing.JoinableQueue()
    
    for author in get_generator("SELECT DISTINCT authorID FROM author_favorites"): jobqueue.put(author[0])
    print "Finished adding to queue, starting threads"
    stop = multiprocessing.Event()
    lock = multiprocessing.RLock()
    
    p = multiprocessing.Process(target=updater, args=(consumequeue, stop, lock))
    p.start()
    p = multiprocessing.Process(target=timer, args=(timequeue,stop, lock))
    p.start()
    for i in range(10):
    
        p = multiprocessing.Process(target=processer, args=(jobqueue, consumequeue, timequeue, lock))
        p.start()
              
    jobqueue.join()
    print "Waiting for consumer to finish"
    consumequeue.join()
    stop.set()
    print "DONE"




    

