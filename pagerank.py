import sqlite3



c = conn.cursor()

c.execute("CREATE TABLE links (node int, outnode int, outcount int PRIMARY KEY (node, outnode))")
c.execute("CREATE TABLE outcounts (node int, count int)")
c.execute("SELECT storyID, COUNT(storyID) FROM author_favorites GROUP BY storyID"); vals = c.fetchall()
c.executemany("INSERT INTO outcounts VALUES (?,?)", vals)
c.execute("SELECT COUNT(*) FROM outcounts"); length = c.fetchone()[0]

c.execute("CREATE TABLE names (name int)")
c.executemany("INSERT INTO names VALUES (?)", [(x[0],) for x in vals])

for author in get_generator("SELECT DISTINCT authorID FROM author_favorites"):
    cur = author[0]
    favs = [x[0] for x in get_generator("SELECT DISTINCT storyID FROM author_favorites WHERE authorID = %d" % cur)]
    for a, b in combinations(favs, 2):
        c.execute("SELECT count FROM outcounts WHERE node == %d" % a); acount = c.fetchall()[0][0]
        c.execute("SELECT count FROM outcounts WHERE node == %d" % b); bcount = c.fetchall()[0][0]
        try:
            c.execute("INSERT INTO links VALUES (?, ?, ?)", (a, b, acount))
            c.execute("INSERT INTO links VALUES (?, ?, ?)", (b, a, bcount))
        except Exception as e:
            q = 0
    
c.execute("CREATE TABLE ranks (node int, rank float)")

for curr in get_generator("SELECT * FROM names"):
    c.execute("INSERT INTO ranks VALUES (?,?)", (curr[0], 1./length))
    
delta = 1
n_iterations = 0

while delta > epsilon:
    c.execute("CREATE TABLE newranks (node int, rank float)")
    for curr in get_generator("SELECT * FROM names"):
        curr = curr[0]
        rank = (1. - damping) / length
        temp = 0
        for inlink in get_generator("SELECT * FROM links WHERE outnode = %d" % curr):
            c.execute("SELECT * FROM ranks WHERE node == %d" % inlink[0])
            inrank = c.fetchone()[1]
            temp += inrank / inlink[2]
        rank += damping * temp
        c.execute("INSERT INTO newranks VALUES (?,?)", (curr, rank))
    c.execute("SELECT SUM(ABS(newranks.rank - ranks.rank)) FROM newranks JOIN ranks ON newranks.node = ranks.node")
    delta = c.fetchone()[0]
    c.execute("DROP TABLE ranks")
    c.execute("ALTER TABLE newranks RENAME TO ranks")
    n_iterations += 1

def get_generator(statement):
    genc = conn.cursor()
    genc.execute(statement)
    ret = genc.fetchone()
    while ret is not None:
        yield ret
        ret = genc.fetchone()
        
    