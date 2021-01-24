import json
import os
import time
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy

# Enter Twitter API Keys
access_token = '1352703664713052161-Zd0hO2IBh45G081ZAuPDKkxkcf3AOn'#"ENTER ACCESS TOKEN"
access_token_secret = 'OkZ4wMSZ61ieue46Lat5Ve5Ha6skaAqFyj8K9rBFprUJy' #"ENTER ACCESS TOKEN SECRET"
consumer_key = 'S8VooVgIjBeX065l7aM9nrXFh' #"ENTER CONSUMER KEY"
consumer_secret = 'NA2gYfho2CgyF82yGlxifMDarMaAMVwmjxyGrbRBql9kaEBH0X' #"ENTER CONSUMER SECRET"

#evantual tweets array
tweets = []
amount = 0
food4search = 'Beverages'
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
            with open("TwitterData_Stream/" + food4search + "/result.json", 'a') as file:
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
        time.sleep(60 * 2)
        stream.disconnect()
        time.sleep(5)


def main():
    try:
        os.mkdir("TwitterData_Stream")
    except:
        pass
    with open("checked_targets.json", "r") as file:
        list = json.load(file)


    for category in list:
        tracklist = []
        if category == food4search:
            try:
                path = "TwitterData_Stream/" + food4search
                os.mkdir(path)
            except:
                pass
            for target in list[category]:
                tracklist.append(target)
                Twitter_Stream_handler(tracklist)

if __name__ == '__main__':
    main()
