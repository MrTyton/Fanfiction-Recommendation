import urllib2
from .fanfictionClasses import *
import chardet

def scrapePage(page_URL, id):
    req = urllib2.Request(page_URL)
    try:
        response = urllib2.urlopen(req)
    except:
        return None
    if response.getcode() == 200:
        try:
            page = response.read()
            encoding = chardet.detect(page)
            page = page.decode(encoding['encoding'], errors='ignore')
            page = page.encode("ascii", errors='ignore')
        except:
            response.close()
            return None
        if "User is no longer an active member." in page or "User does not exist or is no longer an active member." in page:
            response.close()
            return None
    else:
        response.close()        
        return None
    response.close()
    writtenStart = "<div class='z-list mystories"
    favoriteStart = "<div class='z-list favstories'"
    endString = "</div></div></div>"
    numWritten = page.count(writtenStart)
    numFavorite = page.count(favoriteStart)
    
    if numWritten == 0 and numFavorite == 0: return None
    
    authorfind = """<span style="font-weight:bold;letter-spacing:1px;font-size:18px;font-family:'Georgia','Times New Roman','Times', Sans-serif;">"""
    authorName = page[page.index(authorfind)+len(authorfind)+1:page.find("</span>", page.index(authorfind))]
    
    
    author = Author(id, authorName)
    
    prevIndex = 0
    for i in range(numWritten):
        start = page.index(writtenStart, prevIndex)
        end = page.index(endString, start)
        prevIndex = end
        exert = page[start:end]
        exert = exert.replace("<>", "")
        datatitles = exert[exert.index("data-title"):exert.index("data-wordcount")]
        datatitles = datatitles.replace("<", "").replace(">", "").replace("\"", "").replace("data-title=", "data-title=\"") +"\""
        exert = exert[:exert.index("data-title"):] + datatitles + exert[exert.index("data-wordcount"):]
        author.add_story(Story(exert))
        
    prevIndex = 0
    for i in range(numFavorite):
        start = page.index(favoriteStart, prevIndex)
        end = page.index(endString, start)
        prevIndex = end
        exert = page[start:end]
        exert = exert.replace("<>", "")
        exert = exert.replace("\"\"", "\"")
        datatitles = exert[exert.index("data-title"):exert.index("data-wordcount")]
        datatitles = datatitles.replace("<", "").replace(">", "").replace("\"", "").replace("data-title=", "data-title=\"") +"\""
        exert = exert[:exert.index("data-title"):] + datatitles + exert[exert.index("data-wordcount"):]
        author.add_favorite(Story(exert))
    
        
    return author
        
