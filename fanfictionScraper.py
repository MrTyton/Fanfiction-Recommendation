import time
from .scrapePage import scrapePage
from .SimpleProgress import SimpleProgress
import pickle
import threading
from os.path import isfile
from Queue import Queue
import sys
import sqlite3
from time import sleep, time
authors = {}

class scrapeThread(threading.Thread):
    def __init__(self, startnumber, perthread):
        threading.Thread.__init__(self)
        self.startnumber = startnumber
        self.endnumber = startnumber + perthread
        with sqlite3.connect("fanfiction.db") as conn:
            c = conn.cursor()
            try:
                c.execute("SELECT id FROM authors")
                temp = c.fetchall()
                temp = sorted([x for y in temp for x in y])
                temp = [x for x in temp if x < self.endnumber and x > self.startnumber]
                if temp != []:
                    self.startnumber = max(temp)+1
                print "Starting from %d" % self.startnumber
            except Exception as e:
                print "Something broke: ", e
        
    def run(self):
        fault = 0
        try:
            for i in range(self.startnumber, self.endnumber):
                fault = i
                #print i
                insertion = scrapePage("https://www.fanfiction.net/u/%d" % i, i)
                #print insertion
                if insertion is not None:
                    queue.put(insertion)
                    print "Added %d, %s" % (i, insertion.name)
                    #time.sleep(2)
        except Exception as e:
            with open("output.txt", "a") as fp:
                fp.write("Thread %d on item %d did not complete with exception %s\n\n" % (self.startnumber, fault, str(e)))
        print "Exiting thread %d" % self.startnumber
        
class workerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
        
    def run(self):
        with sqlite3.connect("fanfiction.db") as conn:
            c = conn.cursor()
            try:
                c.execute("CREATE TABLE authors (name string, id int PRIMARY KEY)")
                c.execute("CREATE TABLE author_favorites (authorID int, storyID int)")
                c.execute("CREATE TABLE author_written (authorID int, storyID int)")
                c.execute("CREATE TABLE stories (id int PRIMARY KEY, authorID int, name string, wordcount int, published int, updated int, reviews int, chapters int, completed boolean, category string, rate int, summary string)")
                c.execute("CREATE TABLE story_tags (storyid int, tag string)")
            except Exception as e:
                print "Tables already exist"
            #should add tags, rating, should also probably add reviews
            conn.commit()
            count = 0
            while True:
                author = queue.get()
                try:
                    c.execute("INSERT INTO authors VALUES (?, ?)", (author.name, author.id))
                    c.executemany("INSERT INTO author_favorites VALUES (?, ?)", [(author.id, x) for x in author.favorites])
                    c.executemany("INSERT INTO author_written VALUES (?, ?)", [(author.id, x) for x in author.stories])
                    for authorlist in (author.stories,):#, author.favorites):
                        for key in authorlist.keys():
                            curr = authorlist[key]
                            try:
                                c.execute("INSERT INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (curr.ID, curr.authorID, curr.title, curr.wordcount, curr.published, curr.updated, curr.reviews, curr.chapters, 1 if curr.completed else 0, curr.category, curr.rating, curr.summary))
                                if curr.tags != ["None"]:
                                    c.executemany("INSERT INTO story_tags VALUES (?,?)", [(curr.ID, x) for x in curr.tags])
                            except Exception as ez:
                                print "Something broke with story %s" % curr, ez
                except Exception as e:
                    print "Something broke with author %s" % author
                
                print "Processed %s" % author
                if queue.empty():
                    conn.commit()
                queue.task_done()
                if queue.empty():
                    sleep(60)
        
        

#total number: 7077300, 3200
#threadLock = threading.Lock()
threads = []
perthread = 250000
queue = Queue(5000)
workingThread = workerThread()
workingThread.start()
for i in range(1000000, 2000000, perthread):
    addThread = scrapeThread(i, perthread)
    threads.append(addThread)
    
starttime = time()
for curThread in threads:
    curThread.start()
    
for curThread in threads:
    curThread.join()
print "Done, took %d seconds" % (time() - starttime)
queue.join()
sys.exit(1)    


