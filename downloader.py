import json
import os
import time

# Import the necessary methods from tweepy library
from pymongo import MongoClient
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
            with open("TwitterData/result.json", 'a') as file:
                file.write(str(tweet) + '\n')
            file.close()
            tweets.append(tweet)
            print(tweet["created_at"])
            amount += 1
            # time.sleep(10)
            return True


    def on_error(self, status):
        print(status)

def Twitter_Stream_handler(item):
    # Handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    while True:
        stream.filter(track=[item], languages=['en'], is_async=True)
        time.sleep(20)
        stream.disconnect()
    # with open("TwitterData/" + item + ".json", 'a') as file:
    #     json.dump(tweets,file)
    # stream.disconnect()

# def main():
#     try:
#         os.mkdir("TwitterData")
#     except:
#         pass
#     client = MongoClient('mongodb://example:example@sbbi-panda.unl.edu:27017/')
#     db = client.food
#     recipes = db.recipe
#
#     error_recipe_IDs = []
#     error_number = 0
#
#     for recipe in recipes.find():
#         try:
#             recipe_name = str(recipe['name'])
#             print(recipe_name)
#             Twitter_Stream_handler(recipe_name)
#
#         except:
#             error_recipe_IDs.append(recipe['recipe_ID'])
#             error_number += 1
#             pass
#         time.sleep(30)
def Twitter_Cursor_handler():
    processed_tweets = []
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)

    target = 'China Coronavirus'
    text_query = target + ' -filter:retweets'
    tweets = tweepy.Cursor(api.search,q=text_query, lang='en', ).items(100)
    for tweet in tweets:
        print(tweet)
        tweet_dict = {}
        tweet_dict["created_at"] = str(tweet.created_at)
        tweet_dict["id"] = tweet.id_str
        tweet_dict["text"] = tweet.text
        tweet_dict["geo"] = tweet.geo
        tweet_dict["coordinates"] = str(tweet.coordinates)
        tweet_dict["favorite_count"] = tweet.favorite_count
        tweet_dict["entities"] = tweet.entities

        user_info = {}
        user_info["id_str"] = tweet.user.id_str
        user_info["name"] = tweet.user.name
        user_info["screen_name"] = tweet.user.screen_name
        user_info["location"] =  tweet.user.location
        user_info["description"] = tweet.user.description
        user_info["followers_count"] = tweet.user.followers_count
        user_info["friends_count"] = tweet.user.friends_count
        tweet_dict["user"] =user_info
        processed_tweets.append(tweet_dict)

    with open("TwitterData/" + target + ".json", 'w') as file:
        json.dump(processed_tweets,file)

if __name__ == '__main__':
    #main()
    # Twitter_Stream_handler('hot dog')
    # print(amount)
    Twitter_Cursor_handler()