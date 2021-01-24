import json
import os
import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy

# Enter Twitter API Keys
access_token = '1128838733451735042-PWb9rEVvAgBIwhRcjtbaU5FjaGXhS9'#"ENTER ACCESS TOKEN"
access_token_secret = '3CcYPI5sCOzYrgmFTk34bf9FrJAOL9TBGzIfFKv5IcNDg' #"ENTER ACCESS TOKEN SECRET"
consumer_key = '9a8ukdEtOwFPi2Q102wXHY6fl' #"ENTER CONSUMER KEY"
consumer_secret = 'Z1f78bOrK264BLVJKlGLeb85KACGrbwmNOnyew82GWUdnrEl2D' #"ENTER CONSUMER SECRET"

#evantual tweets array
tweets = []
amount = 0

# Create the class that will handle the tweet stream
class StdOutListener(StreamListener):


    def on_data(self, data):
        global tweets
        global amount
        tweet = {}
        data_json = json.loads(data)
        if not data_json['retweeted'] and 'RT @' not in data_json['text']:
            tweet["created_at"] = data_json["created_at"]
            tweet["id"] = data_json["id"]
            tweet["text"] = data_json["text"]
            tweet["user"] = data_json["user"]
            tweet["geo"] = data_json["geo"]
            tweet["coordinates"] = data_json["coordinates"]
            tweet["place"] = data_json["place"]
            tweet["favorite_count"] = data_json["favorite_count"]
            tweet["entities"] = data_json["entities"]
            with open("TwitterData_Stream/result.json", 'a') as file:
                json.dump(tweet, file)
                file.write(",\n")
            file.close()
            tweets.append(tweet)
            print(tweet["created_at"])
            amount += 1
            time.sleep(3)
            return True


    def on_error(self, status):
        print(status)

def Twitter_Stream_handler(tracklist):
    # Handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    while True:
        stream.filter(track=tracklist, languages=['en'], is_async=True)
        time.sleep(60)
        stream.disconnect()



def main():
    try:
        os.mkdir("TwitterData_Stream")
    except:
        pass
    with open("checked_targets.json", "r") as file:
        list = json.load(file)


    for category in list:
        tracklist = []
        try:
            path = "TwitterData/" + category
            os.mkdir(path)
        except:
            pass
        for target in list[category]:
            tracklist.append(target)
            Twitter_Stream_handler(tracklist)

if __name__ == '__main__':
    main()
