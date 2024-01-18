import argparse
import requests as re
import json
import pandas as pd
import time

api_key = ["7IFG7RL02JEWPBA7","VQHCIV3OMT3D7L6K","2GON27Y5N00CVWHG","KKZUA3R00P9MQSG9","W1BJVYKAAT0FERNJ"]
nasdaq100 = pd.read_csv("data/company_info/nasdaq100.csv")

def get_single_news(active_ticker, apidx):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={active_ticker}&apikey={api_key[apidx]}&limit=1000'
    response = re.get(url)
    print(response.status_code)
    with open(f'{active_ticker}.json', 'w+') as f:
        json.dump(response.json(), f)

def get_news_tickers(begin, end, apidx):
    x = nasdaq100[begin:end]
    for i in x['Symbol']:
        time.sleep(5)
        get_single_news(i, apidx)
        print(f'got {i}')
    res = re.get('https://api.ipify.org?format=json')
    print(res.json())
    print("DONE")

def get_news_tickers_list(tickers, apidx):
    for ticker in tickers:
        time.sleep(5)
        get_single_news(ticker, apidx)
        print(f'got {ticker}')
    res = re.get('https://api.ipify.org?format=json')
    print(res.json())
    print("DONE")

def main():
    parser = argparse.ArgumentParser(description='News Data CLI: Note there is a request limit of 25 per day per api key, use vpn when you change keys')
    parser.add_argument('--tickers', nargs='+', help='List of tickers')
    parser.add_argument('--begin', type=int, help='Begin index')
    parser.add_argument('--end', type=int, help='End index')
    parser.add_argument('--apidx', type=int, help='API key index')

    args = parser.parse_args()

    if args.tickers:
        get_news_tickers_list(args.tickers, args.apidx)
    elif args.begin and args.end and args.apidx:
        get_news_tickers(args.begin, args.end, args.apidx)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()





