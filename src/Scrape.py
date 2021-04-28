import requests

# Read API key file
# try:
#     with open("api_key.txt") as keyfile:
#         apikey = keyfile.readline()
# except:
apikey = "AIzaSyCDP3US7ZdB92l3UYvVri4SEdqfkDppovs"  #hardcoded for now

def getvideobyid(id):
    url = "https://youtube.googleapis.com/youtube/v3/videos"
    payload = {'part' : 'id,statistics,snippet', 'id' : id, 'key' : apikey} #id is for verification (not used), statistics is most counts, snippet is for category id
    head = {'Accept' : 'application/json'}
    response = requests.get(url,params=payload,headers=head)
    # print(response.url)
    return response

def getparamsfromjson(jresponse):
    # print(jresponse)
    items = jresponse.get('items')[0]   #for some reason items is a list with the dict at index 0
    stats = items['statistics']         #everything else from the json is nested dicts
    params = list()                     #this should be a PD data frame? or we can do that later
    params.append(items['snippet']['categoryId'])   #'params' is in the same order as 'preNorm' for whatever it's worth
    fields = ['viewCount','likeCount','dislikeCount','commentCount']
    for var in fields:
        params.append(stats[var])
    # print(params)
    return params

def getdatafromid(id):
    response = getvideobyid(id)
    params = getparamsfromjson(response.json())
    #see notes in __main__ for todo list
    return params

def getdatafromurl(url):
    vid = url.split('=')[-1]
    return getdatafromid(vid)
    
if __name__ == "__main__":
    #testing
    url = "https://www.youtube.com/watch?v=QqsLTNkzvaY" #trending science video (Kurzgesagt)
    params = getdatafromurl(url)
    # params are in right order, but it's not a matching dataframe
    # TO DO: Make it a data frame?
    # normalize it
    # split it to y,x
    print(params)
