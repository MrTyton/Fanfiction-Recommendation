import time
from Fanfiction.scrapePage import scrapePage, scrapeReview
#from .SimpleProgress import SimpleProgress
from Fanfiction.fanfictionClasses import Author, Review
import pickle
#import threading
import multiprocessing
from os.path import isfile
#from Queue import Queue
import sys
import sqlite3
from time import sleep, time
from copy import deepcopy
import random
import pickle
import os.path
authors = {}

class scrapeThread(multiprocessing.Process):
    """def __init__(self, startnumber, perthread):
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
                print "Something broke: ", e"""
    
    def __init__(self, i, userqueue, consumerqueue):
        multiprocessing.Process.__init__(self)
        self.startnumber = i
        self.userqueue = userqueue
        self.consumerqueue = consumerqueue
        #self.tohit = deepcopy(numbers)
        
    def run(self):
        #for i in range(self.startnumber, self.endnumber):
        #for i in self.tohit:
        while not self.userqueue.empty():
            i = self.userqueue.get()
            for x in range(3):
                try:
                    #print i
                    insertion = scrapePage("https://www.fanfiction.net/u/%d" % i, i)
                    #print insertion
                    if insertion is not None:
                        self.consumerqueue.put(insertion)
                        print "Added %d, %s" % (i, insertion.name)
                        self.userqueue.task_done()
                        break
                        #time.sleep(2)
                    if insertion is None:
                        self.userqueue.task_done()
                        break
                except Exception as e:
                    if x == 3:
                        with open("output.txt", "a") as fp:
                            fp.write("Thread %d on item %d broke, with exception: %s\n\n" % (self.startnumber, i, str(e)))
                        self.userqueue.task_done()
        print "Exiting thread %d" % self.startnumber
        
class workerThread(multiprocessing.Process):
    def __init__(self, consumerqueue, jobqueue, stop, startrest):
        multiprocessing.Process.__init__(self)
        self.consumerqueue = consumerqueue
        self.jobqueue = jobqueue
        self.stop = stop
        self.startrest = startrest
        
        
    def run(self):
        with sqlite3.connect("fanfiction.db") as conn:
            c = conn.cursor()
            try:
                c.execute("CREATE TABLE authors (name string, id int PRIMARY KEY)")
                c.execute("CREATE TABLE author_favorites (authorID int, storyID int)")
                c.execute("CREATE TABLE author_written (authorID int, storyID int)")
                c.execute("CREATE TABLE stories (id int PRIMARY KEY, authorID int, name string, wordcount int, published int, updated int, reviews int, chapters int, completed boolean, category string, rate int, language string, summary string)")
                c.execute("CREATE TABLE story_tags (storyid int, tag string)")
                c.execute("CREATE TABLE reviews (storyid int, chapter int, reviewer int, content string)")
            except Exception as e:
                print "Tables already exist", e
            #should add tags, rating, should also probably add reviews
            conn.commit()
            self.startrest.set()
            count = 0
            while not self.stop.is_set():
                while not self.consumerqueue.empty():
                    item = self.consumerqueue.get()
                    if isinstance(item, Author):
                        author = item
                        try:
                            c.execute("INSERT INTO authors VALUES (?, ?)", (author.name, author.id))
                            c.executemany("INSERT INTO author_favorites VALUES (?, ?)", [(author.id, x) for x in author.favorites])
                            c.executemany("INSERT INTO author_written VALUES (?, ?)", [(author.id, x) for x in author.stories])
                            for authorlist in (author.stories, author.favorites):
                                for key in authorlist.keys():
                                    curr = authorlist[key]
                                    try:
                                        c.execute("INSERT INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (curr.ID, curr.authorID, curr.title, curr.wordcount, curr.published, curr.updated, curr.reviews, curr.chapters, 1 if curr.completed else 0, curr.category, curr.rating, curr.language, curr.summary))
                                        if curr.tags != ["None"]:
                                            c.executemany("INSERT INTO story_tags VALUES (?,?)", [(curr.ID, x) for x in curr.tags])
                                        if curr.reviews != 0:
                                            self.jobqueue.put(curr.ID)
                                    except Exception as ez:
                                        print "Something broke with story %s" % curr, ez
                        except Exception as e:
                            print "Something broke with author %s" % author, e
                        print "Processed %s" % author
                    elif isinstance(item, list):
                        rev = item
                        try:
                            c.executemany("INSERT INTO reviews VALUES (?,?,?,?)", [(curr.storyID, curr.chapter, curr.user, curr.review) for curr in rev])
                        except Exception as e:
                            print "Something broke with review for %d" % rev[0].storyID, e
                        print "Processed reviews for %d" % rev[0].storyID
                    elif item is None: a=0
                    else:
                        print "Wtf did you pass me"
                    self.consumerqueue.task_done()
                conn.commit()
                sleep(60)
        
        
class reviewScrape(multiprocessing.Process):
    def __init__(self, jobqueue, consumerqueue, stop):
        multiprocessing.Process.__init__(self)
        self.jobqueue = jobqueue
        self.consumerqueue = consumerqueue
        self.stop = stop
        
    def run(self):
        while not self.stop.is_set():
            while not self.jobqueue.empty():
                storyid = self.jobqueue.get()
                try:
                    reviews = scrapeReview(storyid)
                    if reviews != []:
                        self.consumerqueue.put(reviews)
                    print "Added reviews for %d" % storyid
                except Exception as e:
                    with open("output.txt", "a") as fp:
                        fp.write("Review thread broke on storyid %d, with exception: %s\n" % (storyid, str(e)))
                self.jobqueue.task_done()
            

#total number: 7077300, 3200
#threadLock = threading.Lock()
<<<<<<< HEAD
threads = []
perthread = 2500
queue = Queue(5000)
workingThread = workerThread()
workingThread.start()
for i in range(2001000, 2011000, perthread):
    addThread = scrapeThread(i, perthread)
    threads.append(addThread)
=======
if __name__ == "__main__":
    perthread = 20000
    consumerqueue = multiprocessing.JoinableQueue()
    jobqueue = multiprocessing.JoinableQueue()
    userqueue = multiprocessing.JoinableQueue()
    startrest = multiprocessing.Event()
    stop = multiprocessing.Event()
>>>>>>> origin/master
    
    if os.path.isfile("numbers.pkl"):
        print "Loading numbers from file"
        with open("numbers.pkl", "r") as fp:
            numbers = pickle.load(fp)
        print len(numbers)
        #numbers = numbers[990000:]
        if os.path.isfile("fanfiction.db"):
            with sqlite3.connect("fanfiction.db") as conn:
                print "Calculating starting number"
                c = conn.cursor()
                c.execute("SELECT id FROM authors")
                authors = [x[0] for x in c.fetchall()]
                startval = max([numbers.index(x) if numbers.count(x) != 0 else -1 for x in authors])
                print "Starting from %d, %f percent of the way" % (startval, float(startval)/len(numbers))
                with open("output.txt", "a") as fp:
                    fp.write("Starting from %d, %f percent of the way\n" % (startval, float(startval)/len(numbers)))
                numbers = numbers[startval+1:]
            print "Determining reviews"
            c.execute("SELECT id FROM stories WHERE id NOT IN (SELECT DISTINCT storyid FROM reviews) AND reviews != 0"); revs = c.fetchall()
            print "Adding %d stories to the review queue" % len(revs)
            with open("output.txt", "a") as fp:
                fp.write("Adding %d stories to the review queue\n" % len(revs))
            revs = [x[0] for x in revs]
            for x in revs: jobqueue.put(x)
    else:
        numbers = random.sample(xrange(int(7e6)), int(1e6))
        with open("numbers.pkl", "w") as fp:
            pickle.dump(numbers, fp)
    for x in numbers: userqueue.put(x)
    
    
    workingThread = workerThread(consumerqueue, jobqueue, stop, startrest)
    workingThread.start()
    startrest.wait()

    starttime = time()
    for i in range(5):
        addThread = scrapeThread(i, userqueue, consumerqueue)
        addThread.start()
        #threads.append(addThread)
    for i in range(5):
        addThread = reviewScrape(jobqueue, consumerqueue, stop)
        addThread.start()
    #for curThread in threads:
    #    curThread.join()
    userqueue.join()
    with open("output.txt", "a"):
        fp.write("Finished processing users\n")
    for i in range(5):
        addThread = reviewScrape(jobqueue, consumerqueue, stop)
        addThread.start()
    jobqueue.join()
    consumerqueue.join()
    stop.set()
    print "Done, took %d seconds, waiting for all threads to exit." % (time() - starttime)
    sys.exit(1)    


