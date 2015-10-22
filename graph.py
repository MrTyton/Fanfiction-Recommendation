from itertools import combinations
import pickle
from .SimpleProgress import SimpleProgress
from scipy.sparse import bsr_matrix
import numpy as np
import sqlite3

class Node():
    def __init__(self, name):
        self.name = name
        self.vertexis = []#{}
        #self.weight = 0
    
    def update(self, author_favorites):
        self.weight += 1. / (20 + author_favorites)
    
    def add_vertex(self, node, author_favorites):
        nodename = node.name
        #print nodename
        #print [x[0] for x in self.vertexis]
        resolute = [x[0] for x in self.vertexis]
        if nodename in resolute:
            #self.vertexis[nodename] = (self.vertexis[nodename][0] + 1, self.vertexis[nodename][1] +  1. / (20 + author_favorites))
            index = resolute.index(nodename)
            self.vertexis[index][1] += 1
            #self.vertexis[nodename] += 1
        else:
            #self.vertexis[nodename] = (1, 1. / (20 + author_favorites))
            #self.vertexis[nodename] = 0
            self.vertexis.append([nodename, 1])
            
    def __repr__(self):
        return "Story %s, with a weight of %f, connected to %d other nodes" % (self.name, 0, len(self.vertexis))



#when I add a vertex, I put in the weight of uS of a intersect b


class Graph():
    def __init__(self):
        self.nodes = {}
    
    def update_with_author(self, author):
        fav = len(author.favorites)
        for curfav in author.favorites:
            if int(curfav) not in self.nodes:
                self.nodes[int(curfav)] = Node(int(curfav))
            #self.nodes[curfav].update(fav)
        for first, second in combinations(author.favorites, 2):
            first = self.nodes[int(first)]
            second = self.nodes[int(second)]
            first.add_vertex(second, fav)
            second.add_vertex(first, fav)
            
    def __repr__(self):
        return "Fanfiction graph with %d nodes and %d vertixes" % (len(self.nodes), self.calculate_vertixes())
    
    def calculate_vertixes(self):
        total = 0
        for cur in self.nodes:
            total += len(self.nodes[cur].vertexis)
        return total
    

def createGraph(filename):
    with open(filename, "r") as fp:
        authors = pickle.load(fp)
        
    ans = Graph()
    #timer = SimpleProgress(len(authors))
    #timer.start_progress()
    #t = 0
    total = len(authors)
    for i, author in enumerate(authors):
        #print timer.update(t)
        #t += 1
        print "On %d out of %d" % (i, total)
        ans.update_with_author(authors[author])
    
    return ans

def useSparse(authors, stories):
    with open(authors, "r") as fp:
        auth = pickle.load(fp)
    with open(stories, "r") as fp:
        stor = pickle.load(fp)
    
    stordict = {v:i for i, v in enumerate(stories.keys())}
    
    matrix = bsr_matrix((len(stor), len(stor)), dtype=np.int8)
    
    for author in auth:
        curAuthor = auth[author]
        for first, second in combinations(curAuthor.favorites):
            a = 0
            

def reverseAuthors(input, output):
    with open(input, "r") as fp:
        authors = pickle.load(fp)
        
    reversed = {}
    
    for author in authors:
        auth = authors[author]
        weight = 1. / (20 + len(auth.favorites))
        for cur in auth.favorites:
            if cur in reversed:
                reversed[cur].append((author, weight))
            else:
                reversed[cur] = [(author, weight)]
    for cur in reversed:
        reversed[cur] = set(reversed[cur])
    
    with open(output, "w") as fp:
        pickle.dump(reversed, fp)
        
    return
                
    
def create_Sparse_Matrix(input):
    with open(input, "r") as fp:
        data = pickle.load(fp)
    timer = SimpleProgress(56741203756)
    timer.start_progress()
    time = 0
    conn = sqlite3.connect("StoryGraph.db")
    c = conn.cursor()
    for first, second in combinations(data, 2):
        if time % 100000000 == 0:
            conn.commit()
            print timer.update(time)
        time += 1
        s = data[first]
        t = data[second]
        union = s & t
        if len(union) == 0: continue
        
        sum1 = sum([x[1] for x in union])
        sum2 = sum([x[1] for x in s | t])
        
        c.execute("""INSERT INTO map VALUES (?,?,?,?,?,?,?)""", (int(first), int(second), len(s), len(t), len(union), sum1, sum2))
        c.execute("""INSERT INTO map VALUES (?,?,?,?,?,?,?)""", (int(second), int(first), len(t), len(s), len(union), sum1, sum2))
    conn.close()
    

def favorites_Crawl(author):
    favorites = [int(x) for x in author.favorites]
    
    
    #c.execute("""CREATE TABLE map (first_id int, second_id int, num_favorites_1 int, num_favorites_2 int, num_favorites_both int, weight_user_instersection float, weight_user_union float)""")