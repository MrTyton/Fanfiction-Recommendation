from os import listdir
from os.path import isfile, join
import pickle

def compileStories(directory):
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    files = ["%s/%s" % (directory, x) for x in files]
    stories = {}
    authors = {}
    for cur in files:
        with open(cur, "r") as fp:
            currentAuthors = pickle.load(fp)
        for i in currentAuthors:
            currentAuthor = currentAuthors[i]
            for stor in currentAuthor.stories:
                if stor in stories:
                    if currentAuthor.stories[stor].updated > stories[stor].updated:
                        stories[stor] = currentAuthor.stories[stor]
                else:
                    stories[stor] = currentAuthor.stories[stor]
            for stor in currentAuthor.favorites:
                if stor in stories:
                    if currentAuthor.favorites[stor].updated > stories[stor].updated:
                        stories[stor] = currentAuthor.favorites[stor]
                else:
                    stories[stor] = currentAuthor.favorites[stor]
            currentAuthor.stories = currentAuthor.stories.keys()
            currentAuthor.favorites = currentAuthor.favorites.keys()
            authors[currentAuthor.id] = currentAuthor
    return authors, stories
        