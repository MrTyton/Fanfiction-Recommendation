from __future__ import division

from HTMLParser import HTMLParser
import logging
import os
import random
import sqlite3
import sys

from gensim import corpora

from fanfictionClasses import openStoryPage
from topic import Topic


__all__ = []
__version__ = 0.97
__date__ = '2015-11-20'
__updated__ = '2015-11-25'

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d.strip())
    def get_data(self):
        return ' '.join(self.fed)

class StoryBOWCorpus():
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def __iter__(self):    
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        logging.info("loading ids from database")
        c=conn.execute("SELECT a.id, count(f.storyID) FROM authors a, author_favorites f WHERE f.authorID=a.id GROUP BY a.id HAVING count(f.storyID)>4")
        #readers = random.shuffle([row[0] for row in c])[:100]
        readers = [row[0] for row in c]
        c.close()
        logging.info("loading stories from {} readers".format(len(readers)))
        c=conn.execute("SELECT id, chapters FROM stories WHERE language='English'")
        logging.info("loading text from the, um, inter-net")
        t=Topic("T")
        for reader in readers:
            c=conn.execute("SELECT chapters FROM stories WHERE id=?", int(reader))
            if row[0] is not None:
                story_text = get_story_text(int(reader), int(row[0]))
                bow = self.dictionary.doc2bow(t.tokenize(story_text))
                yield bow

class StoryFavCorpus():
    def __init__(self, minfavs=50):
        self.minfavs = minfavs
        
    def __iter__(self):    
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        logging.info("loading ids from database")
        c=conn.execute("SELECT a.id, count(f.storyID) FROM authors a, author_favorites f WHERE f.authorID=a.id GROUP BY a.id HAVING count(f.storyID)>{}".format(self.minfavs-1))
        #readers = random.shuffle([row[0] for row in c])[:100]
        readers = []
        storycount = 0
        for row in c:
            readers.append(row[0]) 
            storycount +=int(row[1])
        
        c.close()
        logging.info("loading {} stories from {} readers".format(storycount, len(readers)))
        #c=conn.execute("SELECT id, chapters FROM stories WHERE language='English'")
        logging.info("loading text from the, um, inter-net")
        random.shuffle(readers)
        for reader in readers:
            logging.info("authorID={}".format(reader))
            c=conn.execute("SELECT f.storyID, s.chapters, s.language FROM author_favorites f, stories s WHERE f.authorID={} AND s.id=f.storyID".format(int(reader)))
            for row in c:
                
                if row[2] is not None and row[2]!='English':
                    logging.debug("Skipping non-English ({}) story ID {}".format(row[2], row[0]))
                    continue
                if row[0] is not None:
                    yield [row[0],row[1]]


def get_story_text(storyID, chapters=10):
    chapter_texts = []
    for i in range(chapters):
        curChapter = i + 1
        url = "https://www.fanfiction.net/s/%d/%d/" % (storyID, curChapter)
        page = openStoryPage(url)
        if page is None: continue
        try:
            start = page.index("<div class='storytext xcontrast_txt nocopy' id='storytext'>")
            end = page.index("</div>\n</div>", start)
            html = page[start+len("<div class='storytext xcontrast_txt nocopy' id='storytext'>"):end]
            text = strip_tags(html)
            chapter_texts.append(text)
        except Exception: continue
    return ' '.join(chapter_texts)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def get_connection():
    return sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")

def scrape_story_texts(storyIDs=None):
    basedir = "/export/apps/dev/fanfiction/stories"
    logging.info("Creating corpus to generate stories as bags of words")
    corpus = StoryFavCorpus()
    storycount=0
    for [sid, chapters] in corpus:
        idpfx = '{}'.format(sid)[:2]
        storydir = '{}/{}'.format(basedir, idpfx)
        try:
            os.mkdir(storydir)
        except OSError:
            pass
        storycount=storycount+1
        if os.path.exists('{}/{}.txt'.format(storydir, sid)):
            logging.debug("Skipping {}, already exists".format(sid))
        else:
            logging.info("Writing storyID={}".format(sid))
            storytext = get_story_text(int(sid), int(chapters))
            with open('{}/{}.txt'.format(storydir, sid), 'w') as storyout:
                storyout.write('{}\n'.format(storytext))
        if storycount % 100 ==0 :
            logging.info("{} stories written".format(storycount))

def scrape_stories(storyIDs=None):
    basedir = "/export/apps/dev/fanfiction"
    logging.info("Loading dictionary from file")
    dictionary = corpora.Dictionary.load("{}/models/summaries_1p.dict".format(basedir))
    logging.info("Creating corpus to generate stories as bags of words")
    corpus = StoryBOWCorpus(dictionary)
    logging.info("Writing stories as corpus of bags of words to disk")
    fname='{}/models/story_bags_{}.mm'.format(basedir, __version__)
    corpora.MmCorpus.serialize(fname, corpus, progress_cnt=1)
         
def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p', level=logging.INFO)
    scrape_story_texts()
    #logging.info("STORY TEXT: {}".format(get_story_text(10921726, 5)))
    logging.error("GOOD BYE!")

main()

    
