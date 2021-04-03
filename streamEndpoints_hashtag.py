import threading
import json
import os
import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream


# Enter Twitter API Keys
from streamEndpoints import StdOutListener

access_token = '1352703664713052161-Zd0hO2IBh45G081ZAuPDKkxkcf3AOn'#"ENTER ACCESS TOKEN"
access_token_secret = 'OkZ4wMSZ61ieue46Lat5Ve5Ha6skaAqFyj8K9rBFprUJy' #"ENTER ACCESS TOKEN SECRET"
consumer_key = 'S8VooVgIjBeX065l7aM9nrXFh' #"ENTER CONSUMER KEY"
consumer_secret = 'NA2gYfho2CgyF82yGlxifMDarMaAMVwmjxyGrbRBql9kaEBH0X' #"ENTER CONSUMER SECRET"

class StdOutListener(StreamListener):
    def on_data(self, data):
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
            with open("user_group/result.json", 'a') as file:
                json.dump(tweet, file)
                file.write(",\n")
            file.close()
            print(tweet["created_at"])
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
        time.sleep(50)

if __name__ == '__main__':
    Twitter_Stream_handler(tracklist=['#fatboyproblems', '#fatgirlproblems'])