'''
CLASS: Web Scraping with Beautiful Soup

What is web scraping?
- Extracting information from websites (simulates a human copying and pasting)
- Based on finding patterns in website code (usually HTML)

What are best practices for web scraping?
- Scraping too many pages too fast can get your IP address blocked
- Pay attention to the robots exclusion standard (robots.txt)
- Let's look at http://www.imdb.com/robots.txt

What is HTML?
- Code interpreted by a web browser to produce ("render") a web page
- Let's look at example.html
- Tags are opened and closed
- Tags have optional attributes

How to view HTML code:
- To view the entire page: "View Source" or "View Page Source" or "Show Page Source"
- To view a specific part: "Inspect Element"
- Safari users: Safari menu, Preferences, Advanced, Show Develop menu in menu bar
- Let's inspect example.html
'''

# read the HTML code from a web page and save as a string
with open('example.html', 'rU') as f:
    html = f.read()

# convert HTML into a structured Soup object
from bs4 import BeautifulSoup
b = BeautifulSoup(html)

# print out the object
print b
print b.prettify()
type(b)

# 'find' method returns the first matching Tag (and everything inside of it)
b.find(name='body')
b.find(name='h1')
b.find(name='title')

# Tags allow you to access the 'inside text'
b.find(name='h1').text

# Tags also allow you to access their attributes
b.find(name='h1')['id']

# 'find_all' method is useful for finding all matching Tags
b.find(name='p')        # returns a Tag
b.find_all(name='p')    # returns a ResultSet (like a list of Tags)

# ResultSets can be sliced like lists
len(b.find_all(name='p'))
b.find_all(name='p')[0]
b.find_all(name='p')[0].text
b.find_all(name='p')[0]['id']

# iterate over a ResultSet
results = b.find_all(name='p')
for tag in results:
    print tag.text

# limit search by Tag attribute
b.find(name='p', attrs={'id':'scraping'})
b.find_all(name='p', attrs={'class':'topic'})
b.find_all(attrs={'class':'topic'})

b.find_all(attrs={'class':'topic'})[0].text

topics = b.find_all(attrs={'class':'topic'})
for row in topics:
    print(row.text)


# limit search to specific sections
b.find_all(name='li')
b.find(name='ul', attrs={'id':'scraping'}).find_all(name='li')



'''
EXERCISE ONE
'''

# find the 'h2' tag and then print its text
b.find(name='h2').text


# find the 'p' tag with an 'id' value of 'feedback' and then print its text
b.find_all(name='p', attrs={'id':'feedback'})


# find the first 'p' tag and then print the value of the 'id' attribute
b.find(name='p')['id']


# print the text of all four resources displayed on the webpage
results = b.find_all(name='ul')
for tag in results:
    print tag.text


# print the text of only the API resources
results = b.find_all(name='ul', attrs={'id':'api'})
for tag in results:
    print tag.text

'''
Scraping the IMDb website
'''

# get the HTML from the Shawshank Redemption page
import requests

r = requests.get('http://www.imdb.com/title/tt0111161/')
#r = requests.get('http://www.imdb.com/title/tt0133093/') # The Matrix

# convert HTML into Soup
b = BeautifulSoup(r.text)
print b


# write text from b.  Encode the text as 'utf8'
import codecs


# writing with the default of ASCII causes a write failure due to utf8 characters in b
f = codecs.open(r'/Users/reneehosogi/Documents/GitHub_Clones/GA-SEA-DAT1/data/Shawshank.html', "w", 'utf-8')
f.write (b.prettify()) 
f.close()


# Uncomment and then run the next 3 lines only you have encoding errors
# import sys
#reload(sys)
# sys.setdefaultencoding('utf8')

# get the title
b.find_all(name='span', attrs={'class':'itemprop', 'itemprop':'name'})    # too many results
b.find(name='span', attrs={'class':'itemprop', 'itemprop':'name'}).text   # just get the first
b.find(name='h1').find(name='span', attrs={'class':'itemprop', 'itemprop':'name'}).text   # limit the search

# get the star rating (as a float)
float(b.find(name='span', attrs={'itemprop':'ratingValue'}).text)
float(b.find(name='div', attrs={'class':'titlePageSprite star-box-giga-star'}).text)

'''
EXERCISE TWO
'''

# get the description
b.find(name='p', attrs={'itemprop':'description'}).text

    
# get the content rating
b.find(name='meta', attrs={'itemprop':'contentRating'})['content']

# get the duration in minutes (as an integer)
int(b.find(name='time', attrs={'itemprop':'duration'}).text.strip())[-2:1]
int(b.find(name='time', attrs={'itemprop':'duration'})['datetime']



'''
OPTIONAL WEB SCRAPING HOMEWORK

First, define a function that accepts an IMDb ID and returns a dictionary of
movie information: title, star_rating, description, content_rating, duration.
The function should gather this information by scraping the IMDb website, not
by calling the OMDb API. (This is really just a wrapper of the web scraping
code we wrote above.)

For example, get_movie_info('tt0111161') should return:

{'content_rating': 'R',
 'description': u'Two imprisoned men bond over a number of years...',
 'duration': 142,
 'star_rating': 9.3,
 'title': u'The Shawshank Redemption'}

Then, open the file imdb_ids.txt using Python, and write a for loop that builds
a list in which each element is a dictionary of movie information.

Finally, convert that list into a DataFrame.
'''

# define a function that accepts an IMDb ID and returns a dictionary of movie information
import requests

r = requests.get('http://www.imdb.com/title/tt0458352/')

from bs4 import BeautifulSoup
b = BeautifulSoup
b = BeautifulSoup(r.text)

print b

import codecs

import seaborn as sns

# test the function
print r

# open the file of IDs (one ID per row), and store the IDs in a list


# get the information for each movie, and store the results in a list


# check that the list of IDs and list of movies are the same length


# convert the list of movies into a DataFrame


'''
Another IMDb example: Getting the genres
'''

# read the Shawshank Redemption page again
r = requests.get('http://www.imdb.com/title/tt0111161/')
b = BeautifulSoup(r.text)

# only gets the first genre
b.find(name='span', attrs={'class':'itemprop', 'itemprop':'genre'})

# gets all of the genres
b.find_all(name='span', attrs={'class':'itemprop', 'itemprop':'genre'})

# stores the genres in a list
[tag.text for tag in b.find_all(name='span', attrs={'class':'itemprop', 'itemprop':'genre'})]

'''
Another IMDb example: Getting the writers
'''

# attempt to get the list of writers (too many results)
b.find_all(name='span', attrs={'itemprop':'name'})

# limit search to a smaller section to only get the writers
b.find(name='div', attrs={'itemprop':'creator'}).find_all(name='span', attrs={'itemprop':'name'})

'''
Another IMDb example: Getting the URLs of cast images
'''

# find the images by size
results = b.find_all(name='img', attrs={'height':'44', 'width':'32'})

# check that the number of results matches the number of cast images on the page
len(results)

# iterate over the results to get all URLs
for tag in results:
    print tag['loadlate']

'''
Useful to know: 
When writing to a file, Python 2.7x by default writes it with ASCII encoding.
Writing with the default of ASCII will cause a write failure if there are any utf8 characters in the text being written.
Use a codecs.open() to write the file with utf8 encoding.
#
'''
import codecs
f = codecs.open(r'/Users/jim_byers/Documents/GA/GA_Data_Science_course/SEA-DAT1/data/Shawshank.html', "w", 'utf-8')
f.write (b.prettify()) 
f.close()


'''
Useful to know:
Alternative Beautiful Soup syntax
'''

# read the example web page again
with open('example.html', 'rU') as f:
    html = f.read()

# convert to Soup
b = BeautifulSoup(html)

# these are equivalent
b.find(name='p')    # normal way
b.find('p')         # 'name' is the first argument
b.p                 # can also be accessed as an attribute of the object

# these are equivalent
b.find(name='p', attrs={'id':'scraping'})   # normal way
b.find('p', {'id':'scraping'})              # 'name' and 'attrs' are the first two arguments
b.find('p', id='scraping')                  # can write the attributes as arguments

# these are equivalent
b.find(name='p', attrs={'class':'topic'})   # normal way
b.find('p', class_='topic')                 # 'class' is special, so it needs a trailing underscore
b.find('p', 'topic')                        # if you don't name it, it's assumed to be the class

# these are equivalent
b.find_all(name='p')    # normal way
b.findAll(name='p')     # old function name from Beautiful Soup 3
b('p')                  # if you don't name the method, it's assumed to be find_all
