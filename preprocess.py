import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer


def process_dataset(json_path, train_rate):
    # 从原始json文件提取有用信息，划分为训练集、验证集、测试集
    print('## Read the json file...')
    df = pd.read_json(json_path, lines=True)
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['userID', 'itemID', 'review', 'rating']
    # 将用户/物品id映射为数字
    df['userID'] = df.groupby(df['userID']).ngroup()
    df['itemID'] = df.groupby(df['itemID']).ngroup()

    with open('data/embedding/stopwords.txt') as f:  # 停用词
        stop_words = set(f.read().splitlines())
    with open('data/embedding/punctuations.txt') as f:  # 无用标点
        punctuations = set(f.read().splitlines())

    def clean_review(review):  # 清洗文本
        review = review.lower()
        for p in punctuations:
            review = review.replace(p, ' ')  # 替换标点
        review = WordPunctTokenizer().tokenize(review)  # 分词
        review = [word for word in review if word not in stop_words]  # 删除停用词
        # review = [nltk.WordNetLemmatizer().lemmatize(word) for word in review]  # 词干提取
        return ' '.join(review)

    df['review'] = df['review'].apply(clean_review)  # 清洗文本
    df = df.drop(df[[not isinstance(x, str) or len(x) == 0 for x in df['review']]].index)  # 清除空评论, **很重要**！

    print(f'## Got {len(df)} reviews from json! Split them into train,validation and test!')
    train, valid = train_test_split(df, test_size=1 - train_rate, random_state=3)  # 数据集划分，含乱序
    valid, test = train_test_split(valid, test_size=0.5, random_state=4)
    print(f'## Saving the data. count: train {len(train)}, valid {len(valid)}, test {len(test)}')
    train.to_csv(os.path.dirname(json_path) + '/train.csv', index=False, header=False)
    valid.to_csv(os.path.dirname(json_path) + '/valid.csv', index=False, header=False)
    test.to_csv(os.path.dirname(json_path) + '/test.csv', index=False, header=False)
    return train, valid, test


if __name__ == '__main__':
    print('## preprocess.py: Begin to load the data...')
    start_time = time.perf_counter()
    process_dataset('data/music/Digital_Music_5.json', train_rate=0.8)
    end_time = time.perf_counter()
    print(f'## preprocess.py: Data loading complete! Time used {end_time - start_time:.0f} seconds.')
