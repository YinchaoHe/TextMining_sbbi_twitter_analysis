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



def Twitter_Cursor_handler(target, path):
    processed_tweets = []
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)

    text_query = target + ' -filter:retweets'
    c = tweepy.Cursor(api.search,q=text_query, lang='en').items()

    target = target.replace(" ", "_")
    count = 0
    while count <= 2000:
        try:
            tweet = c.next()
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
            user_info["location"] = tweet.user.location
            user_info["description"] = tweet.user.description
            user_info["followers_count"] = tweet.user.followers_count
            user_info["friends_count"] = tweet.user.friends_count
            tweet_dict["user"] = user_info
            processed_tweets.append(tweet_dict)
            count += 1
            print(tweet_dict)
            with open(path + "/" + target + ".json", 'a') as file:
                json.dump(tweet_dict, file)
                file.write(",\n")
            file.close()
            time.sleep(5)
        except tweepy.TweepError:
            time.sleep(60 * 2)
            continue
        except StopIteration:
            break

    # for tweet in tweets:
    #     print(tweet)
    #     tweet_dict = {}
    #     tweet_dict["created_at"] = str(tweet.created_at)
    #     tweet_dict["id"] = tweet.id_str
    #     tweet_dict["text"] = tweet.text
    #     tweet_dict["geo"] = tweet.geo
    #     tweet_dict["coordinates"] = str(tweet.coordinates)
    #     tweet_dict["favorite_count"] = tweet.favorite_count
    #     tweet_dict["entities"] = tweet.entities
    #
    #     user_info = {}
    #     user_info["id_str"] = tweet.user.id_str
    #     user_info["name"] = tweet.user.name
    #     user_info["screen_name"] = tweet.user.screen_name
    #     user_info["location"] =  tweet.user.location
    #     user_info["description"] = tweet.user.description
    #     user_info["followers_count"] = tweet.user.followers_count
    #     user_info["friends_count"] = tweet.user.friends_count
    #     tweet_dict["user"] =user_info
    #     processed_tweets.append(tweet_dict)



# def read_target_file():
#     file =  open("search_targets.txt", 'r')
#     check_categories = []
#     original_categories = []
#
#     for line in file.readlines():
#         text = line.split('\n')[0]
#         target_food = text.split()[0:1][0]
#         category = text.split()[-1]
#         pair_t = [target_food, category]
#         original_categories.append(pair_t)
#         if category not in check_categories:
#             check_categories.append(category)
#
#     file.close()
#     print(len(check_categories))
#     print(len(original_categories))
#     cate = {}
#
#     for category in check_categories:
#         cate[category] = []
#         for pair in original_categories:
#             if pair[1] == category:
#                 cate[category].append(pair[0])
#     with open("targets.json", 'w') as file:
#         json.dump(cate,file)
#     file.close()

def main():
    try:
        os.mkdir("TwitterData_Cursor")
    except:
        pass
    with open("checked_targets.json", "r") as file:
        list = json.load(file)
    for category in list:
        try:
            path = "TwitterData_Cursor/" + category
            os.mkdir(path)
        except:
            pass
        for target in list[category]:
            print(target)
            Twitter_Cursor_handler(target, path)

if __name__ == '__main__':
    main()

