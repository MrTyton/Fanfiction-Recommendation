import urllib2
from .fanfictionClasses import *
import chardet
from time import sleep
from BeautifulSoup import BeautifulSoup

def scrapePage(page_URL, id):
    req = urllib2.Request(page_URL)
    for i in range(3):
        try:
            response = urllib2.urlopen(req, timeout=10)
        except:
            if i == 2: return None
            sleep(10)
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
    authfindindex = page.index(authorfind)
    authorName = page[authfindindex+len(authorfind)+1:page.find("</span>", authfindindex)]
    
    
    author = Author(id, authorName)
    
    prevIndex = 0
    
    index = page.index
    
    a_s = author.add_story
    a_f = author.add_favorite
    
    for i in range(numWritten):
        start = index(writtenStart, prevIndex)
        end = index(endString, start)
        prevIndex = end
        exert = page[start:end]
        exert = exert.replace("<>", "")
        a = exert.index("data-title")
        b = exert.index("data-wordcount")
        datatitles = exert[a:b]
        datatitles = "%s\"" % datatitles.replace("<", "").replace(">", "").replace("\"", "").replace("data-title=", "data-title=\"")
        exert = "%s%s%s" % (exert[:a],datatitles,exert[b:])
        a_s(Story(exert))
        
    prevIndex = 0
    for i in range(numFavorite):
        start = index(favoriteStart, prevIndex)
        end = index(endString, start)
        prevIndex = end
        exert = page[start:end]
        exert = exert.replace("<>", "").replace("\"\"", "\"")
        a = exert.index("data-title")
        b = exert.index("data-wordcount")
        datatitles = exert[a:b]
        datatitles = "%s\"" % datatitles.replace("<", "").replace(">", "").replace("\"", "").replace("data-title=", "data-title=\"")
        exert = "%s%s%s" % (exert[:a],datatitles,exert[b:])
        a_f(Story(exert, False))
    
        
    return author

def openReviewPage(url):
    req = urllib2.Request(url)
    for i in range(3):
        try:
            response = urllib2.urlopen(req, timeout=10)
        except:
            if i == 2: return None
            sleep(10)
    if response.getcode() == 200:
        try:
            page = response.read()
            encoding = chardet.detect(page)
            page = page.decode(encoding['encoding'], errors='ignore')
            page = page.encode("ascii", errors='ignore')
        except:
            print "Excepted"
            response.close()
            return None
        if page.find("Story not found.") != -1 and page.find("No Reviews found.") != -1:
            print "Nothing here"
            response.close()
            return None
    else:
        print "Response code fuckup"
        response.close()        
        return None
    response.close()
    return page

def scrapeReview(storyID):
    reviews = []
    
    page = openReviewPage("http://www.fanfiction.net/r/%d/0/1" % storyID)
    
    if page == None: return None
    
    check = page.find("/'>Last</a>")
    if check != -1:
        totalPages = int(page[page.rfind("/", 0, check)+1:check])
    else:
        totalPages = 1
    #print totalPages
    page = None
    for q in range(totalPages):
        
        while page == None:
            page = openReviewPage("http://www.fanfiction.net/r/%d/0/%d" % (storyID, q+1))
        
        
        end = page.find("id='gui_table1i'")
        end = page.find("tbody", end)
        numReviews = page.count("<tr  >", end)
        #print end
        for i in range(numReviews):
            start = page.find("<tr  >", end)
            end = page.find("</tr>", start)
            #print start
            #print end
            exert = page[start:end]
            #print exert
            reviews.append(Review(exert, storyID))
            
    return reviews
        
    