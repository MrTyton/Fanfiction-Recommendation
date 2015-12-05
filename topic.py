from __future__ import division

import logging
import os
import pickle
import random
import sqlite3
import string
import time
import traceback

from gensim import corpora, models
import lda
from nltk import word_tokenize
import nltk
import numpy as np
from scipy import spatial
from fanfictionClasses import openStoryPage
from _sqlite3 import Cursor
from sympy.mpmath.calculus.extrapolation import fold_finite

__all__ = []
__version__ = 0.98
__date__ = '2015-11-20'
__updated__ = '2015-12-03'

class Topic():
    def __init__(self, basedir):
        self.basedir = basedir
        self.load_stopwords()

    def load_stopwords(self):
        self.stopwords = ['...','(',')',',','.','?','!','"','\'', ';', ':','\'s','``','\'\'','n\'t','\'re','\'d','\'m','au']
        self.stopwords.extend(nltk.corpus.stopwords.words('english'))
        self.stopwords.extend(string.punctuation)
        stopwordsfile = self.basedir+'/stopwords.txt'
        with open(stopwordsfile, 'r') as stopin:
            self.stopwords.extend([x.strip() for x in stopin])
        
    def tokenize(self, text):
        return word_tokenize(text.lower())
    
    def text_to_vector(self, text):
        story_word_counts = [0 for word in self.vocab]
        words = self.tokenize(text)
        wordcount=0 
        newvocab=[]
        for word in words:
            if word in self.stopwords:
                continue
            try:
                idx = self.vocab.index(word)
                wordcount=wordcount+1
                if idx<0 or idx>= len(story_word_counts):
                    logging.error("Vocab index {} is not valid for story_word_counts len={}".format(idx,len(story_word_counts)))
                else:
                    story_word_counts[idx] =story_word_counts[idx] +1 
            except Exception:
                #logging.warn("Unknown word '{}'".format(word))
                
                newvocab.add(word)
        return (story_word_counts, newvocab)
    
    
    def run_lda_on_summaries(self, N=10000, storyids=None):
        conn = sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.basedir))
        if storyids is None:
            logging.info("Connected")
            c=conn.execute("SELECT id FROM stories WHERE language='English'")
            logging.info("Executed")
            storyids = [a[0] for a in c]
            c.close()
            #stories = [(a[0],a[1]) for a in c]
            logging.info("Collected {} story ids".format(len(storyids)))
        if len(storyids)>N:
            random.shuffle(storyids)
            logging.info("Shuffled")
            #logging.warn("Confused?")
            storyids = storyids[:N]
            logging.info("Sampled")
        #for row in c:
        # Store counts of words per story in a matrix
        # We don't have the vocab to start with, so it grows
        #stories = stories[:N]
        thematrix = []
        vocab = []
        storymap = dict()
        storyindex=0
        #for [storyid, summary] in stories:
        starttime = time.time()
        minutes = 0
        for storyid in storyids:
            story_word_counts = []
            if storyindex==0:
                logging.info("Creating matrix from {} story summaries".format(N))
            ctime = time.time()
            try:
                summaries_read = len(thematrix)
                if ((ctime-starttime)//60)>minutes:
                    minutes = (int)((ctime-starttime)//60)
                    logging.info("{} summaries read after {} minutes.".format(summaries_read, minutes))
                elif summaries_read % 1000 == 0:
                    logging.info("{} summaries read after {:.3f} seconds".format(summaries_read, ctime-starttime))
                for _ in vocab:
                    story_word_counts.append(0)
                c2 = conn.execute("SELECT (name || ' ' || summary) as abstract FROM stories WHERE id={}".format(storyid))
                for row in c2:
                    summary = row[0].strip()
                c2.close()
                if not summary:
                    continue
                summary=summary.strip()
                (story_word_counts, newvocab) = self.text_to_vector(summary)
                self.vocab.extend(newvocab)
                nzcounts = [x for x in story_word_counts if x>0]
                if len(nzcounts)>0:
                    storymap[storyindex] = storyid
                    storyindex=storyindex+1
                    thematrix.append(story_word_counts)
            except Exception as e:
                logging.error("ERROR (storyid={}): {}".format(storyid, e))
                logging.error(traceback.format_exc())
        conn.close()
        return (thematrix, vocab, storymap)

    def load_nparray(self, thematrix, vocab):
        logging.info("Initializing NP array")
        #stories = []
        X = np.zeros((len(thematrix), len(vocab)), dtype=np.intc)
        n=0
        logging.info("Loading NP array from the matrix")
        for story_word_counts in thematrix:
            for _ in range(len(vocab)-len(story_word_counts)):
                story_word_counts.append(0)
            for i in range(len(vocab)):
                X[n, i] = story_word_counts[i]
            n=n+1
        return X
    
    def write_dimensions(self, vocab, storymap):
        with open("vocab.txt", 'w') as vout:
            for term in vocab:
                vout.write("{}\n".format(term))
        with open("storymap.txt", 'w') as sout:
            for i in range(len(storymap)):
                sout.write("{},{}\n".format(i,storymap[i]))
                
    def fit_model(self, thematrix, vocab, storymap, n_topics=50, n_iter=500, alpha=0.1, eta=0.01):
        X = self.load_nparray(thematrix, vocab)
        logging.error("X=({})".format(X.shape))
        model = lda.LDA(n_topics=n_topics, n_iter=n_iter, alpha=alpha, eta=eta, random_state=1)
        logging.info("Initialized LDA model, alpha={}, eta={}".format(alpha, eta))
        model.fit(X)
        logging.info("Fit LDA model to X")
        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 20
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            logging.info('Topic {}: {}'.format(i, ' '.join(topic_words)))
        doc_topic = model.doc_topic_
        for i in range(10):
            logging.info("{} (top topic: {})".format(storymap[i], doc_topic[i].argmax()))
        logging.info("DONE!")
        modelfile ="X_v{}_n{}_a{}_eta{}_k{}_i{}.model".format(__version__, len(thematrix),alpha,eta,n_topics,n_iter) 
        with open(modelfile, 'w') as modelout:
            pickle.dump(model, modelout)
        logging.info("Model written to {}".format(modelfile))

class TopicModelExperiment():
    def __init__(self, name, modelfile="/export/apps/dev/workspace/Fanfiction/X_n14994_a0.1_eta0.01_k25_i500.model", vocabfile="/export/apps/dev/workspace/Fanfiction/vocab.txt"):
        self.name = name
        self.load_stopwords()
        self.load_model(modelfile)
        self.load_vocab(vocabfile)
        
    def tokenize(self, text):
        return word_tokenize(text.lower())
    def load_stopwords(self):
        self.stopwords = ['...','(',')',',','.','?','!','"','\'', ';', ':','\'s','``','\'\'','n\'t','\'re','\'d','\'m','au']
        self.stopwords.extend(nltk.corpus.stopwords.words('english'))
        self.stopwords.extend(string.punctuation)
        with open('/export/apps/dev/fanfiction/stopwords.txt', 'r') as stopin:
            self.stopwords.extend([x.strip() for x in stopin])
        
    def load_model(self, modelfile):
        with open(modelfile, 'r') as modelin:
            self.model = pickle.load(modelin)
    
    def load_vocab(self, vocabfile):
        with open(vocabfile, 'r') as vocabin:
            self.vocab=[]
            for line in vocabin:
                self.vocab.append(line.strip())
    
    def text_to_vector(self, text):
        story_word_counts = [0 for word in self.vocab]
        words = self.tokenize(text)
        wordcount=0 
        newvocab=[]
        for word in words:
            if word in self.stopwords:
                continue
            try:
                idx = self.vocab.index(word)
                wordcount=wordcount+1
                if idx<0 or idx>= len(story_word_counts):
                    logging.error("Vocab index {} is not valid for story_word_counts len={}".format(idx,len(story_word_counts)))
                else:
                    story_word_counts[idx] =story_word_counts[idx] +1 
            except Exception:
                #logging.warn("Unknown word '{}'".format(word))
                
                newvocab.add(word)
        return (story_word_counts, newvocab)
    
    
    def evaluate_model(self):
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        logging.info("Connected")
        c=conn.execute("SELECT a.id, count(f.storyID) FROM authors a, author_favorites f WHERE f.authorID=a.id GROUP BY a.id HAVING count(f.storyID)>4")
        logging.info("Executed")
        readers = [row[0] for row in c]
        c.close()
        c=conn.execute("SELECT id FROM stories WHERE language='English'")
        storyids = random.shuffle([row[0] for row in c])
        
        for reader in readers:
            c=conn.execute("SELECT f.storyID, (s.name || ' ' || s.summary) as abstract FROM author_favorites f, stories s WHERE s.id=f.storyID AND f.authorID=?", int(reader))
            thematrix = []
            for row in c:
                #storyid = row[0]
                summary = row[1].strip()
                story_word_counts=self.text_to_vector(summary)[0]
                thematrix.add(story_word_counts)
                c2 = conn.execute("SELECT (name || ' ' || summary) as abstract FROM stories WHERE id=?", storyids[0])
                storyids = storyids[1:]
                for row2 in c2:
                    summary = row2[0].strip()
                    if summary:
                        story_word_counts= self.text_to_vector()[0]
                        thematrix.add(story_word_counts)
# Want to collect fav and non-fav in equal proportions, 
# then separate into two clusters using k-means. 
# Success is % correct separation of stories into fav/non-fav                    
            #topicmatrix = self.reduce_matrix(thematrix)
            # OK now I have a topic matrix of favorites for this reader

    '''
      Writes story topic proportions to file name provided (or 'story_topics.csv'). 
      Requires access to sqlite datbase. 
      Also writes 'newvocab.txt' or newvocabfile for any vocabulary not already in 
      self.vocab. 
    ''' 
    def write_topics_to_file(self, storytopicsfile=None, newvocabfile=None):
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        logging.info("Connected")
        c=conn.execute("SELECT id, (name || ' ' || summary) as abstract FROM stories WHERE language='English'")
        logging.info("Executed")
        if( storytopicsfile is None ):
            storytopicsfile = "story_topics.csv" 
        if( newvocabfile is None ):
            newvocabfile = "newvocab.txt"
        vout = open(storytopicsfile, "w")
        storycount = 0
        thematrix = []
        thestories = []
        newvocab = set()
        for row in c:
            storyid = row[0]
            summary = row[1].strip()
            (story_word_counts, vocabadds)=self.text_to_vector(summary)
            
            newvocab.intersection_update(set(vocabadds))
            nullstory = True
            for count in story_word_counts:
                if count>0:
                    nullstory=False
                    break
            if nullstory:
                continue
            # Create a one row matrix for this story, then
            # use the model to reduce to topics
            
            thematrix.append(story_word_counts)
            thestories.append(storyid)
            storycount = storycount+1
            if storycount % 5000 == 0 :
                topicmatrix = self.reduce_matrix(thematrix)
                for i in range(len(thestories)):
                    xstoryid = thestories[i]
                    topics = topicmatrix[i]
                    vout.write("{}".format(xstoryid))
                    for t in topics:
                        vout.write(",{:.5f}".format(t))
                    vout.write("\n")
                logging.info("{} stories reduced and written".format(storycount))
                thematrix = []
                thestories = []
        with open(newvocabfile, 'w') as nvout:
            for word in newvocab:
                nvout.write("{}\n".format(word))
        c.close()
        vout.close()
        conn.close()

    def load_nparray(self, thematrix, vocab):
        logging.info("Initializing NP array")
        #stories = []
        X = np.zeros((len(thematrix), len(vocab)), dtype=np.intc)
        n=0
        logging.info("Loading NP array from the matrix")
        for story_word_counts in thematrix:
            for _ in range(len(vocab)-len(story_word_counts)):
                story_word_counts.append(0)
            for i in range(len(vocab)):
                X[n, i] = story_word_counts[i]
            n=n+1
        return X
    
    def reduce_vector(self, story_word_counts):
        return self.reduce_matrix([story_word_counts])

    def reduce_matrix(self, thematrix):
        X=self.load_nparray(thematrix, self.vocab)
        T = self.model.fit_transform(X)
        logging.info("Reduced {} to {}".format(X.shape, T.shape))
        return T
'''
class VariationalEM():
    def __init__(self, name, vocabfile="/export/apps/dev/workspace/Fanfiction/vocab.txt"):
        self.name = name
        self.load_stopwords()
        #self.load_model(modelfile)
        self.load_vocab(vocabfile)
    def is_converging(self, phi, lphi, gamma, lgamma):
        return True
    
    def e_step(self, storyids, n_topics, alpha, phi=None, gamma=None):
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        for storyid in storyids:
            phi = [[1/n_topics for _ in n_topics] for _ in self.topic_model.vocab]
            N = len(self.topic_model.vocab)
            gamma = [alpha+N/n_topics]
            lastphi=None
            lastgamma=None
            while lastphi is None or lastgamma is None or not self.is_converging(phi, lastphi, gamma, lastgamma):
                
                sql = "SELECT summary FROM stories WHERE id=?"
                c=conn.execute(sql, storyid)
                for row in c:
                    summary = row[0].strip()
                
                    #(story_word_counts, newvoc) = self.topic_model.text_to_vector(summary)
                
        if phi is None:
            phi = [[1/n_topics for _ in n_topics] for _ in storyids]
    
    def build_lda_model(self, thematrix, vocab, storymap, N=10000, k=25, iter=500, alpha=0.1, eta=0.01):
        self.topic_model = Topic("Variational EM")
        
        (thematrix, vocab, storymap)=self.topic_model.run_lda_on_summaries(N)
        self.topic_model.write_dimensions(vocab, storymap)
        self.topic_model.fit_model(thematrix, vocab, storymap, n_topics=k, n_iter=iter, alpha=alpha, eta=eta)

'''
class OnlineLDAExperiment():
    def __init__(self, numtopics=150, basedir="/export/apps/dev/fanfiction", modelfile=None):
        self.basedir=basedir
        self.load_stopwords()
        self.numtopics = numtopics
        self.story_vector_cache=dict()
        if modelfile is not None:
            self.lda = models.ldamodel.LdaModel.load(modelfile)

    def prep_for_eval(self, fold, penalty=20):
        self.fold = fold
        self.penalty=penalty
        
    def populate(self,favorites):
        pass
    
    def load_stopwords(self):
        self.stopwords = ['...','(',')',',','.','?','!','"','\'', ';', ':','\'s','``','\'\'','n\'t','\'re','\'d','\'m','au']
        self.stopwords.extend(nltk.corpus.stopwords.words('english'))
        self.stopwords.extend(string.punctuation)
        stopwordsfile = '{}/stopwords.txt'.format(self.basedir) 
        with open(stopwordsfile, 'r') as stopin:
            self.stopwords.extend([x.strip() for x in stopin])
    
    def similarity(self, u, v):
        return 1 - spatial.distance.cosine(u,v)
    
    def tokenize(self, text):
        return [x for x in word_tokenize(text.lower()) if x not in self.stopwords]
    
    def topic_map_to_vector(self, topic_map, k=100):
        story_topic_vector = []
        for i in range(k):
            if i in topic_map:
                story_topic_vector.append(topic_map[i])
            else:
                story_topic_vector.append(0.000)
        return story_topic_vector
    
    def get_topic_vector_for_story(self, storyID):
        if storyID in self.story_vector_cache:
            story_topic_vector = self.story_vector_cache[storyID]
            return story_topic_vector
        conn = get_connection(self.basedir)
        c=conn.execute("SELECT (name || ' ' || summary) as abstract FROM stories WHERE language='English' and id={}".format(storyID))
        if c is None:
            raise Exception("Bad query for storyID = {}".format(id))
        else:
            row = [x for x in c][0]
            summary = row[0].strip()
            if summary:
                story_word_counts = self.dictionary.doc2bow(self.tokenize(summary))
                story_topic_proportions = dict((x,y) for (x,y) in self.lda[story_word_counts])
                story_topic_vector = self.topic_map_to_vector(story_topic_proportions, int(self.numtopics))
                self.story_vector_cache[storyID] = story_topic_vector
            else:
                raise Exception("No summary found for story {}".format(storyID))
        return story_topic_vector
    
    '''
      storyID - ID of story to evaluate
      favorites - list of storyIDs that are favorites for an author (reader)
      
      returns - 0.000 - 1.000, representing likelihood that the story is a favorite, given list of favorites. 
    '''
    def favorite_likelihood(self, storyID, favorites):
        story_topic_vector = self.get_topic_vector_for_story(storyID)
        score = max([self.similarity(story_topic_vector, self.get_topic_vector_for_story(fav)) for fav in favorites])
        return score
        
    def evaluate_model(self, modelfile="/export/apps/dev/fanfiction/models/lda1p_0.93_k5_a1.0_enil.model"):
        logging.info("Loading model from file")
        lda = models.ldamodel.LdaModel.load(modelfile)
        modelfilesfx = "_".join(modelfile.split("/")[-1].split(".model")[0].split("_")[1:])
        k = int(modelfilesfx.split("_")[1][1:])
        logging.info("Suffix = _{}; K={}".format(modelfilesfx, k))
        logging.info("Loading dictionary from file")
        self.dictionary = corpora.Dictionary.load("{}/models/summaries_1p.dict".format(self.basedir))
        if os.path.exists("{}/models/summaries_topics_{}.mm".format(self.basedir, modelfilesfx)):
            logging.info("Loading corpus from disk")
            corpus = corpora.MmCorpus('{}/models/summaries_topics_{}.mm'.format(self.basedir, modelfilesfx))
        elif os.path.exists("{}/models/summaries_1p.mm".format(self.basedir)):
            logging.info("Loading corpus from disk")
            corpus = corpora.MmCorpus('{}/models/summaries_1p.mm'.format(self.basedir))
            logging.info("Writing corpus as topics to disk")
            corpora.MmCorpus.serialize('{}/models/summaries_topics_{}.mm'.format(self.basedir,modelfilesfx), lda[corpus])
            corpus = corpora.MmCorpus('{}/models/summaries_topics_{}.mm'.format(self.basedir, modelfilesfx))
        else:
            logging.info("Creating corpus to generate summaries as bags of words")
            corpus = FanFictionCorpus(self.dictionary)
            logging.info("Writing corpus as topics to disk")
            corpora.MmCorpus.serialize('{}/models/summaries_topics_{}.mm'.format(self.basedir, modelfilesfx), lda[corpus])
            corpus = corpora.MmCorpus('{}/models/summaries_topics_{}.mm'.format(self.basedir, modelfilesfx))
        conn = sqlite3.connect("/media/export/apps/dev/fanfiction/fanfiction_no_reviews.db")
        logging.info("selecting readers with at least 5 favorite stories")
        
        # Select 100 readers  
        # with at least 5 favorites
        c=conn.execute("SELECT a.id, count(f.storyID) FROM authors a, author_favorites f WHERE f.authorID=a.id GROUP BY a.id HAVING count(f.storyID)>4")
        #readers = random.shuffle([row[0] for row in c])[:100]
        readers = [row[0] for row in c][:100]
        c.close()
        mrrs=[]
        logging.info("Evaluating 100 readers")
        for reader in readers:
            Query = "SELECT f.storyID, (s.name || ' ' || s.summary) as abstract FROM author_favorites f, stories s WHERE s.id=f.storyID AND f.authorID={} AND s.language='English'".format(reader)
            c=conn.execute(Query)
            favsummaries = [(row[0],row[1]) for row in c]
            if favsummaries is None:
                logging.error("ERROR: Query failed to return favorites for reader {}".format(reader))
                logging.error("QUERY: {}".format(Query))
                continue
            split = int(len(favsummaries)*0.05)
            if split==0:
                split=1
            heldout = favsummaries[:split]
            train = favsummaries[split:]
            logging.info('Reader {} has {} favorites: {} for train, {} for eval'.format(reader, len(favsummaries), len(train), len(heldout)))
            thematrix = []
            # Create a topic proportion matrix 
            # for all the favorite stories in train
            for row in train:
                #storyid = row[0]
                summary = row[1].strip()
                if summary:
                    story_word_counts = self.dictionary.doc2bow(self.tokenize(summary))
                    story_topic_proportions = dict((x,y) for (x,y) in lda[story_word_counts])
                    story_topic_vector = self.topic_map_to_vector(story_topic_proportions, int(k))
                    
                    thematrix.append(story_topic_vector)
            heldoutvectors = []
            for row in heldout:
                #storyid = row[0]
                summary = row[1].strip()
                if summary:
                    story_word_counts = self.dictionary.doc2bow(self.tokenize(summary))
                    story_topic_proportions = dict((x,y) for (x,y) in lda[story_word_counts])
                    heldoutvectors.append(self.topic_map_to_vector(story_topic_proportions, k))
            # Choose the most similar favorite story
            # to use for ranking
            logging.debug("Scanning 10,000 stories from corpus")
            scores = []
            doccount=0
            for vector in corpus:
                #logging.info("Compare {} to {}".format(vector, thematrix[0]))
                story_topic_proportions = dict((x,y) for (x,y) in vector)
                score = max([self.similarity(self.topic_map_to_vector(story_topic_proportions, k), fav) for fav in thematrix])
                scores.append(score)
                doccount+=1
                if doccount % 10000 == 0 :
                    break
            logging.info("Scan complete; {} story summaries scanned. Sorting scores".format(doccount))
            scores.sort(reverse=True)
            logging.debug("Sorting complete. ")
            # Now check the heldout
            rrarr=[]
            logging.info("Comparing held out set to favorites.")
            for heldoutvec in heldoutvectors:
                evalscores=[self.similarity(heldoutvec, fav) for fav in thematrix]
                score = max(evalscores)
                
                i=0
                while scores[i]>=score and i<len(scores):
                    i=i+1 
                rrarr.append(1.0/(float(i+1)))
            rdrmrr = np.average(rrarr)
            logging.info("Reader {}: MRR={:.5e}".format(reader, rdrmrr))
            mrrs.append(rdrmrr)
        mrr=np.average(mrrs)
        c.close()
        conn.close()
        logging.info("Overall Results (MRR): {:.4e}".format(mrr))
        return mrr
# Want to collect fav and non-fav in equal proportions, 
# then separate into two clusters using k-means. 
# Success is % correct separation of stories into fav/non-fav                    
            #topicmatrix = self.reduce_matrix(thematrix)
            # OK now I have a topic matrix of favorites for this reader

    def run_lda_on_summaries(self, n_topics=50, alpha='auto', eta='auto'):
        if os.path.exists("{}/models/summaries_1p.dict".format(self.basedir)):
            logging.info("Loading dictionary from file")
            self.dictionary = corpora.Dictionary.load("{}/models/summaries_1p.dict".format(self.basedir))
        else:
            conn = sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.basedir))
            
            # Change this if using folds
            logging.info("Selecting summaries from DB")
            c=conn.execute("SELECT (name || ' ' || summary) as abstract FROM stories WHERE language='English'")
            logging.info("Saving vocab to dictionary")
            
            self.dictionary = corpora.Dictionary([[word for word in self.tokenize("{}".format(a[0]))] for a in c])
            c.close()
            conn.close()
            logging.info("Listing words that appear only once")
            once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems() if docfreq == 1]
            logging.info("Filtering dictionary")
            self.dictionary.filter_tokens(once_ids)
            logging.info("Compactifying dictionary")
            self.dictionary.compactify()
            
            #dictionary = corpora.Dictionary(summaries)
            logging.info("Writing dictionary to file")
            self.dictionary.save('{}/models/summaries_1p.dict'.format(self.basedir));
        
        vocabsize = len(self.dictionary.keys())
        if eta is None:
            etastr = 'nil'
        elif eta=='auto':
            etaval = 1/n_topics
            etastr = '{:.3f}'.format(etaval)
            #etaval = 1/n_topics
        
            eta = np.zeros((n_topics, vocabsize))
            for topicidx in range(n_topics):
                for termidx in range(vocabsize):
                    eta[topicidx, termidx] = 1/n_topics
        else:
            eta = float(eta)
            etastr = '{:.3f}'.format(eta) 
        if not alpha=='auto':
            alpha = float(alpha)
        modelfile ="{}/models/lda1p_{}_k{}_a{}_e{}.model".format(self.basedir, __version__, n_topics, alpha, etastr) 
        if os.path.exists(modelfile):
            logging.warn("File exists: {}".format(modelfile))
            logging.warn("I assume you don't want to overwrite it.")
            logging.warn("If in fact you do, please move the existing file out of the way.")
            return modelfile
        if os.path.exists("{}/models/summaries_1p.mm".format(self.basedir)):
            logging.info("Loading corpus from disk")
            corpus = corpora.MmCorpus('{}/models/summaries_1p.mm'.format(self.basedir))
            #corpus = [dictionary.doc2bow(text) for text in summaries]
        else:
            logging.info("Creating corpus to generate summaries as bags of words")
            corpus = FanFictionCorpus(self.dictionary)
            logging.info("Saving corpus to disk")
            corpora.MmCorpus.serialize('{}/models/summaries_1p.mm'.format(self.basedir), corpus)
        
        logging.info("Training model from corpus")
        self.lda = models.ldamodel.LdaModel(corpus=corpus, alpha=alpha, eta=eta, id2word=self.dictionary, num_topics=n_topics, update_every=1, chunksize=20000, passes=1)
        
        logging.info("Saving model to disk")
        self.lda.save(modelfile)
        logging.info("DONE!")
        return modelfile

class StoryTextCorpus():
    def __init__(self, dictionary, basedir="/export/apps/dev/fanfiction"):
        self.dictionary = dictionary
    
    def __iter__(self):    
        conn = sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.basedir))
        logging.info("loading ids from database")
        c=conn.execute("SELECT id, chapters FROM stories WHERE language='English'")
        logging.info("loading text from the, um, inter-net")
        t=Topic("T")
        for row in c:
            if row[0] is not None:
                story_text = get_story_text(row[0], int(row[1]))
                bow = self.dictionary.doc2bow(t.tokenize(story_text))
                yield bow
        
class FanFictionCorpus():
    def __init__(self, dictionary, basedir="/export/apps/dev/fanfiction"):
        self.name = "Fan Fiction Corpus (c) 2015"
        self.dictionary = dictionary
        self.basedir=basedir
        
    def __iter__(self):
        conn = sqlite3.connect("{}/fanfiction_no_reviews.db".format(self.basedir))
        logging.info("loading summaries from database")
        c=conn.execute("SELECT (name || ' ' || summary) as abstract FROM stories WHERE language='English'")
        t = Topic("T")
        for row in c:
            if row[0] is not None:
                summary = ('{}'.format(row[0])).strip()
                bow = self.dictionary.doc2bow(t.tokenize(summary))
                #[w for w in t.tokenize(summary.strip()) if w not in t.stopwords]
                yield bow
            
def get_story_text(storyID, chapters=10):
    chapter_texts = []
    for i in range(chapters):
        curChapter = i + 1
        url = "https://www.fanfiction.net/s/%d/%d/" % (storyID, curChapter)
        page = openStoryPage(url)
        if page is None: continue
        try:
            logging.info("Loading from {}".format(url))
            start = page.index("<div class='storytext xcontrast_txt nocopy' id='storytext'>")
            end = page.index("</div>\n</div>", start)
            text = page[start+len("<div class='storytext xcontrast_txt nocopy' id='storytext'>"):end]
            chapter_texts.append(text)
        except Exception: continue
    return ' '.join(chapter_texts)

def get_connection(basedir="/export/apps/dev/fanfiction"):
    return sqlite3.connect("{}/fanfiction_no_reviews.db".format(basedir))

def scrape_stories(storyIDs=None):
    basedir = "/export/apps/dev/fanfiction"
    logging.info("Loading dictionary from file")
    dictionary = corpora.Dictionary.load("{}/models/summaries_1p.dict".format(basedir))
    logging.info("Creating corpus to generate stories as bags of words")
    corpus = StoryTextCorpus(dictionary)
    logging.info("Writing corpus as topics to disk")
    corpora.MmCorpus.serialize('{}/models/story_bags_{}.mm'.format(basedir, __version__), corpus, progress_cnt=100)
