import time
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def calculate_mse(model, dataloader, device):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            user_reviews, item_reviews, reviews, ratings, user_ids, item_ids = [x.to(device) for x in batch]
            latent, predict = model(user_reviews, item_reviews, user_ids, item_ids)
            mse += F.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # mse of dataloader


class TransNetsDataset(Dataset):
    def __init__(self, data_path, word_dict, config):
        self.word_dict = word_dict
        self.r_count = config.review_count
        self.r_length = config.review_length
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)
        self.null_idx = set()  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        reviews = [self._adjust_review_list([x], 1, self.r_length) for x in df['review']]
        reviews = torch.LongTensor(reviews).view(-1, self.r_length)
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        user_ids = torch.LongTensor(df['userID'].to_list()).view(-1, 1)
        item_ids = torch.LongTensor(df['itemID'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        self.reviews = reviews[[idx for idx in range(reviews.shape[0]) if idx not in self.null_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.null_idx]]
        self.user_ids = user_ids[[idx for idx in range(user_ids.shape[0]) if idx not in self.null_idx]]
        self.item_ids = item_ids[[idx for idx in range(item_ids.shape[0]) if idx not in self.null_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.reviews[idx], self.rating[idx],\
                self.user_ids[idx], self.item_ids[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # For every sample(user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # get reviews without review u for i.
            if len(reviews) < self.lowest_r_count:
                self.null_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.r_count, self.r_length)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_count, r_length):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # Certain count.
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # Certain length of review.
        return reviews

    def _review2id(self, review):  # Split a sentence into words, and map each word to a unique number by dict.
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # word to unique number by dict.
            else:
                wids.append(self.PAD_WORD_idx)
        return wids
