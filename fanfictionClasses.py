from BeautifulSoup import BeautifulSoup

class Story():
    def __init__(self, urlString, full=True):
        soup = BeautifulSoup(urlString)
        divpointer = soup.div
        self.ID = int(divpointer['data-storyid'])
        if not full:
            return
        self.category = divpointer['data-category']
        self.title = divpointer['data-title'].strip()
        self.wordcount = int(divpointer['data-wordcount'])
        self.published = int(divpointer['data-datesubmit'])
        self.updated = int(divpointer['data-dateupdate'])
        self.reviews = int(divpointer['data-ratingtimes'])
        self.chapters = int(divpointer['data-chapters'])
        self.completed = True if divpointer['data-statusid'] == '2' else False
        self.url = soup.a['href']
        loc = soup.text.rfind("Rated: ")
        if "%s - " % self.category in soup.text:
            self.summary = soup.text[:soup.text.index("%s - " % self.category)]
        else:
            self.summary = soup.text[:loc]
        if "  by <a href=" in urlString:
            authorstring = urlString[urlString.index("  by <a href="):urlString.index("</a>", urlString.index("  by <a href="))]
            self.author = authorstring[authorstring.index(">")+1:]
            idstring = urlString[urlString.index("  by <a href=\"/u/"):urlString.index("\">", urlString.index("  by <a href="))]
            self.authorID = int(idstring.split("/")[2])
        
        
        loc = soup.text.rfind("Rated: ")
        self.rating = {"Rated: K":0, "Rated: K+":1, "Rated: T":2, "Rated: M":3}[soup.text[loc:soup.text.index(" -", loc)]]
        loc = soup.text.find(" - ", loc)
        self.language = soup.text[loc+3:soup.text.find("-", loc+3)].strip()
        
        check = soup.text.rfind("- ")
        cutting = soup.text[check:]
        while "Complete" in cutting:
            cutting = soup.text[:check]
            check = cutting.rfind("- ")
            cutting = cutting[check:]
        cutting.strip()
        cutting = cutting[2:]
        if "Published" in cutting or "Updated" in cutting: self.tags = ["None"]
        else:
            cutting = cutting.replace("]", "],")
            if "[" not in cutting:
                self.tags = [x.strip() for x in cutting.split(",")]
            else:
                pairings = []
                while ("[") in cutting:
                    a = cutting.index("[")
                    b = cutting.index("]")
                    insert = cutting[a:b+1]
                    pairings.append(insert)
                    cutting = cutting.replace(insert, "")
                self.tags = [x.strip() for x in cutting.split(",")]
                self.tags.extend(pairings)
                pairings = [x.split(",") for x in pairings]
                pairings = [x[1:-1] for y in pairings for x in y]
                self.tags.extend(pairings)
            self.tags = [x for x in self.tags if x != ""]
               
        
    def __repr__(self):
        if self.title is None: return "Story ID: %d" % self.ID
        return "%s, %d: %s by %s" % (self.ID, self.authorID, self.title, self.author) if self.author is not None else "%s: %s" % (self.ID, self.title,)
        
class Author():
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.stories = {}
        self.favorites = {}
        
    def add_story(self, story):
        story.author = self.name
        story.authorID = self.id
        self.stories[story.ID] = story
        
    def add_favorite(self, story):
        self.favorites[story.ID] = story
        
    def __repr__(self):
        return "%s, %s with %d written stories and %d favorites" % (self.name, self.id, len(self.stories), len(self.favorites))
    
class Review():
    def __init__(self, exert, id):
        soup = BeautifulSoup(exert)
        loc = exert.find("<a href='/u/")
        if loc == -1:
            self.user = -1 
        else:
            self.user = int(exert[loc + 12 : exert.find("/", loc+12)])
        loc = exert.find("<small", loc+12)
        self.chapter = int(exert[loc+34: exert.find(" ", loc+34)])
        self.review = soup.div.text
        self.storyID = id
        
    def __repr__(self):
        return "%d wrote on story %d chapter %d, saying %s" % (self.user, self.id, self.chapter, self.review)