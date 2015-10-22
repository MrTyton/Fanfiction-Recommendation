import time
from .scrapePage import scrapePage
from .SimpleProgress import SimpleProgress
import pickle
import threading
from os.path import isfile
from Queue import Queue
import sys
import sqlite3
authors = {}

class scrapeThread(threading.Thread):
    def __init__(self, startnumber, perthread):
        threading.Thread.__init__(self)
        self.startnumber = startnumber
        self.endnumber = startnumber + perthread
        
    def run(self):
        if isfile("./results/authors-%d.pkl" % self.startnumber): return
        threadAuthor = {}
        try:
            for i in range(self.startnumber, self.endnumber):
                insertion = scrapePage("https://www.fanfiction.net/u/%d" % i, i)
                if insertion is not None:
                    threadAuthor[i] = insertion
                    print "Added %d, %s" % (i, insertion.name)
                    #time.sleep(2)
            with open("./results/authors-%d.pkl" % self.startnumber, "w") as fp:
                pickle.dump(threadAuthor, fp)
        except Exception as e:
            with open("output.txt", "a") as fp:
                fp.write("Thread %d did not complete with exception %s" % (self.startnumber, str(e)))
        
class workerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        with sqlite3.connect("StoryGraph.db") as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE authors (name string, id PRIMARY KEY int)")
            c.execute("CREATE TABLE author_favorites (authorID int, storyID int")
            c.execute("CREATE TABLE author_written (authorID int, storyID int")
            c.execute("CREATE TABLE stories (id PRIMARY KEY int, authorID int, name string, wordcount int, published int, updated int, reviews int, chapters int, completed boolean, category string, summary string)")
            #should add tags, rating, should also probably add reviews
            conn.commit()
            count = 0
            while True:
                author = queue.get()
                count += 1
                c.execute("INSERT INTO authors VALUES (?, ?)", (author.name, author.id))
                c.executemany("INSERT INTO author_favorites VALUES (?, ?)", [(author.id, x) for x in author.favorites])
                c.executemany("INSERT INTO author_written VALUES (?, ?)", [(author.id, x) for x in author.stories])
                for key in author.stories:
                    curr = author.stories[key]
                    try:
                        c.execute("INSERT INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?)", (curr.ID, curr.authorID, curr.published, curr.updated, curr.reviews, curr.chapters, 1 if curr.completed else 0, curr.category, curr.summary))
                    except Exception as e:
                        print "Something broke with story %s" % curr
                if count == 5000:
                    conn.commit()
                    count = 0
                queue.task_done()
        
        

#total number: 7077300, 3200
#threadLock = threading.Lock()
threads = []
perthread = 1000
queue = Queue(5000)
workingThread = workerThread()
workingThread.start()
for i in range(0, 100000, perthread):
    addThread = scrapeThread(i, perthread)
    threads.append(addThread)
for curThread in threads:
    curThread.start()
    
for curThread in threads:
    curThread.join()
queue.join()
print "Done"    

#timer = SimpleProgress(100000-16000)
#timer.start_progress()
"""for i in range(20000, 21000):
    if i % 100 == 0:
        with open("authors.pkl", "r") as fp:
            updator = pickle.load(fp)
        updator.update(authors)
        with open("authors.pkl", "w") as fp:
            pickle.dump(updator, fp)
        authors = {}    
    insertion = scrapePage("https://www.fanfiction.net/u/%d" % i, i)
    if insertion is not None:
        authors[i] = insertion
        print "Added %d, %s" % (i, insertion.name)
    else:
        continue
    #time.sleep(2)

with open("authors.pkl", "r") as fp:
    updator = pickle.load(fp)
updator.update(authors)
with open("authors.pkl", "w") as fp:
    pickle.dump(updator, fp)
authors = {}     
"""
