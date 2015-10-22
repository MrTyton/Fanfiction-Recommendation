from BeautifulSoup import BeautifulSoup

class Story():
    def __init__(self, urlString):
        soup = BeautifulSoup(urlString)
        divpointer = soup.div
        self.category = divpointer['data-category']
        self.ID = int(divpointer['data-storyid'])
        self.title = divpointer['data-title']
        self.wordcount = int(divpointer['data-wordcount'])
        self.published = int(divpointer['data-datesubmit'])
        self.updated = int(divpointer['data-dateupdate'])
        self.reviews = int(divpointer['data-ratingtimes'])
        self.chapters = int(divpointer['data-chapters'])
        self.completed = True if divpointer['data-statusid'] == '2' else False
        self.url = soup.a['href']
        if self.category + " - " in soup.text:
            self.summary = soup.text[:soup.text.index(self.category + " - ")]
        else:
            self.summary = soup.text[:soup.text.index("Rated: ")]
        if "  by <a href=" in urlString:
            authorstring = urlString[urlString.index("  by <a href="):urlString.index("</a>", urlString.index("  by <a href="))]
            self.author = authorstring[authorstring.index(">")+1:]
            idstring = urlString[urlString.index("  by <a href=\"/u/"):urlString.index("\">", urlString.index("  by <a href="))]
            self.authorID = int(idstring.split("/")[2])
        
    def __repr__(self):
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