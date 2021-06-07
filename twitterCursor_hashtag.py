import json
import os
import time

import pandas as pd
from tweepy import OAuthHandler
import tweepy

# Enter Twitter API Keys
access_token = '1128838733451735042-PWb9rEVvAgBIwhRcjtbaU5FjaGXhS9'#"ENTER ACCESS TOKEN"
access_token_secret = '3CcYPI5sCOzYrgmFTk34bf9FrJAOL9TBGzIfFKv5IcNDg' #"ENTER ACCESS TOKEN SECRET"
consumer_key = '9a8ukdEtOwFPi2Q102wXHY6fl' #"ENTER CONSUMER KEY"
consumer_secret = 'Z1f78bOrK264BLVJKlGLeb85KACGrbwmNOnyew82GWUdnrEl2D' #"ENTER CONSUMER SECRET"



def Twitter_Cursor_handler(target, path, file_name):
    processed_tweets = []
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)

    text_query = target + ' -filter:retweets'
    c = tweepy.Cursor(api.search,q=text_query, lang='en').items()

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

            with open(path + "/" + file_name + ".json", 'a') as file:
                json.dump(tweet_dict, file)
                file.write(",\n")
            file.close()
            time.sleep(5)
        except tweepy.TweepError:
            time.sleep(60 * 2)
            print('error')
            continue
        except StopIteration:
            break






def hashtag_monitor():
    data = pd.read_csv('publichealth_v1i2e6_app1.csv', header=0)
    data = data.fillna('MISSING')
    path = 'medical_terms'
    try:
        os.mkdir(path)
    except:
        pass
    for medi_term in data.columns:
        search_target = '('
        for k in data[medi_term]:
            if k != 'MISSING':
                k = str.replace(k, '-', ' ')
                if search_target == '(':
                    search_target = search_target + k
                else:
                    search_target = search_target + ' OR ' + k
        search_target += ')'
        print(search_target)

        # #user should be the disease
        # #search_target should be medical terms
        # #path should be constant
        Twitter_Cursor_handler(search_target, path, medi_term)

if __name__ == '__main__':
    #main()
    hashtag_monitor()

