from BeautifulSoup import BeautifulSoup
from .scrapePage import openStoryPage

class Story():
    def __init__(self, ID, authorID, title, wordcount, published, updated, reviews, chapters, completed, category, rating, language, summary, url="", author="", tags=[], chapter_texts={}):
        self.ID = ID
        self.category = category
        self.title = title
        self.wordcount = wordcount
        self.published = published
        self.updated = updated
        self.reviews = reviews
        self.chapters = chapters
        self.completed = completed
        self.url = url
        self.summary = summary
        self.authorID = authorID
        if author == "": self.author = authorID
        else: self.author = author
        self.rating = rating
        self.language = language
        self.tags = tags
        self.chapter_texts = chapter_texts
    
    @classmethod
    def fromURL(cls, urlString, full=True):
        soup = BeautifulSoup(urlString)
        divpointer = soup.div
        ID = int(divpointer['data-storyid'])
        if not full:
            return
        category = divpointer['data-category']
        title = divpointer['data-title'].strip()
        wordcount = int(divpointer['data-wordcount'])
        published = int(divpointer['data-datesubmit'])
        updated = int(divpointer['data-dateupdate'])
        reviews = int(divpointer['data-ratingtimes'])
        chapters = int(divpointer['data-chapters'])
        completed = True if divpointer['data-statusid'] == '2' else False
        url = soup.a['href']
        loc = soup.text.rfind("Rated: ")
        if "%s - " % category in soup.text:
            summary = soup.text[:soup.text.index("%s - " % category)]
        else:
            summary = soup.text[:loc]
        if "  by <a href=" in urlString:
            authorstring = urlString[urlString.index("  by <a href="):urlString.index("</a>", urlString.index("  by <a href="))]
            author = authorstring[authorstring.index(">")+1:]
            idstring = urlString[urlString.index("  by <a href=\"/u/"):urlString.index("\">", urlString.index("  by <a href="))]
            authorID = int(idstring.split("/")[2])
        
        
        loc = soup.text.rfind("Rated: ")
        rating = {"Rated: K":0, "Rated: K+":1, "Rated: T":2, "Rated: M":3}[soup.text[loc:soup.text.index(" -", loc)]]
        loc = soup.text.find(" - ", loc)
        language = soup.text[loc+3:soup.text.find("-", loc+3)].strip()
        
        check = soup.text.rfind("- ")
        cutting = soup.text[check:]
        while "Complete" in cutting:
            cutting = soup.text[:check]
            check = cutting.rfind("- ")
            cutting = cutting[check:]
        cutting.strip()
        cutting = cutting[2:]
        if "Published" in cutting or "Updated" in cutting: tags = ["None"]
        else:
            cutting = cutting.replace("]", "],")
            if "[" not in cutting:
                tags = [x.strip() for x in cutting.split(",")]
            else:
                pairings = []
                while ("[") in cutting:
                    a = cutting.index("[")
                    b = cutting.index("]")
                    insert = cutting[a:b+1]
                    pairings.append(insert)
                    cutting = cutting.replace(insert, "")
                tags = [x.strip() for x in cutting.split(",")]
                tags.extend(pairings)
                pairings = [x.split(",") for x in pairings]
                pairings = [x[1:-1] for y in pairings for x in y]
                tags.extend(pairings)
            tags = [x for x in tags if x != ""]
            
        chapter_texts = {}
        
        return cls(self, ID, authorID, title, wordcount, published, updated, reviews, chapters, completed, category, rating, language, summary, url, author, tags, chapter_texts)
        
        
    def __repr__(self):
        if self.title is None: return "Story ID: %d" % self.ID
        return "%s, %d: %s by %s" % (self.ID, self.authorID, self.title, self.author) if self.author is not None else "%s: %s" % (self.ID, self.title,)
    
    def get_chapter_text(self):
        for i in range(self.chapters):
            curChapter = i + 1
            url = "https://www.fanfiction.net/s/%d/%d/" % (self.ID, curChapter)
            page = openStoryPage(url)
            if page is None: continue
            try:
                start = page.index("<div class='storytext xcontrast_txt nocopy' id='storytext'>")
                end = page.index("</div>\n</div>", start)
                text = page[start+len("<div class='storytext xcontrast_txt nocopy' id='storytext'>"):end]
                self.chapter_texts[curChapter] = text
            except Exception: continue
    
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