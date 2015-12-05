import sqlite3
from random import shuffle

def create_splits(filename):
    with sqlite3.connect(filename) as conn:
        c = conn.cursor()
        c.execute("SELECT authorID FROM author_favorites GROUP BY authorID HAVING COUNT(authorID) > 1 ")
        authors = [x[0] for x in c.fetchall()]
        shuffle(authors)
        num = len(authors)
        train = authors[:int(num * .6)]
        dev = authors[int(num * .6):int(num * .8)]
        test = authors[int(num * .8):]
        
        c.execute("SELECT * FROM author_favorites WHERE authorID IN (" + ','.join((str(n) for n in train)) + ")")
        traindata = c.fetchall()
        c.execute("SELECT authorID, GROUP_CONCAT(storyID) FROM author_favorites WHERE authorID IN (" + ','.join((str(n) for n in dev)) + ") GROUP BY authorID")
        devdata = c.fetchall()
        c.execute("SELECT authorID, GROUP_CONCAT(storyID) FROM author_favorites WHERE authorID IN (" + ','.join((str(n) for n in test)) + ") GROUP BY authorID ORDER BY COUNT(authorID) ASC" % test)
        testdata = c.fetchall()
        
    def split_data(data):
        inputs = []
        targets = []
        for id, values in data:
            values = values.split(",")
            splitnum = int(len(values) * (2/3.))
            if splitnum < 1: splitnum = 1
            elif len(values) - splitnum < 1: splitnum = len(values) - 1 
            shuffle(values)
            for x in values[:splitnum]:
                inputs.append((id, x))
            for x in values[splitnum:]:
                targets.append((id, x))
                
        return inputs, targets
    
    dev_in, dev_tar = split_data(devdata)
    test_in, test_tar = split_data(testdata)
    
    with sqlite3.connect("fanfiction_favorites_splits.db") as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE train_favorites (authorID int, storyID int)")
        c.execute("CREATE TABLE dev_inputs (authorID int, storyID int)")
        c.execute("CREATE TABLE dev_targets (authorID int, storyID int)")
        c.execute("CREATE TABLE test_inputs (authorID int, storyID int)")
        c.execute("CREATE TABLE test_targets (authorID int, storyID int)")
        
        c.executemany("INSERT INTO train_favorites VALUES (?,?)", traindata)
        c.executemany("INSERT INTO dev_inputs VALUES (?,?)", dev_in)
        c.executemany("INSERT INTO dev_targets VALUES (?,?)", dev_tar)
        c.executemany("INSERT INTO test_inputs VALUES (?,?)", test_in)
        c.executemany("INSERT INTO dev_targets VALUES (?,?)", test_tar)
        