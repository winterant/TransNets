import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


class TransNetsDataset(Dataset):
    def __init__(self, data_path, word_dict, config):
        self.word_dict = word_dict
        self.r_count = config.review_count
        self.r_length = config.review_length
        self.PAD_WORD_idx = word_dict[config.PAD_WORD]

        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        df['review'] = df['review'].apply(self._review2id)
        self.null_idx = set()  # Save the indices of empty samples, delete them at last.
        user_reviews = self._get_reviews(df)  # Gather reviews for every user without target review(i.e. u for i).
        item_reviews = self._get_reviews(df, 'itemID', 'userID')
        reviews = [self._adjust_review_list([x], 1, self.r_length) for x in df['review']]
        reviews = torch.LongTensor(reviews).view(-1, self.r_length)
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        self.reviews = reviews[[idx for idx in range(reviews.shape[0]) if idx not in self.null_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.null_idx]]

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # For every sample(user,item), gather reviews for user/item.
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # Information for every user/item
        lead_reviews = []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # get information of lead, return DataFrame.
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # get reviews without review u for i.
            if len(reviews) < 5:
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


class CNN(nn.Module):

    def __init__(self, config, word_dim, review_count=1):
        super(CNN, self).__init__()

        self.kernel_count = config.kernel_count
        self.review_count = review_count

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=word_dim,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),  # out shape(new_batch_size,kernel_count,1)
        )

        self.linear = nn.Sequential(
            nn.Linear(config.kernel_count * self.review_count, config.cnn_out_dim),
            nn.Tanh(),
        )

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        latent = self.conv(vec.permute(0, 2, 1).contiguous())  # output shape(new_batch_size, kernel_count, 1)
        latent = latent.view(-1, self.kernel_count * self.review_count)
        latent = self.linear(latent)
        return latent  # output shape(batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, in_dim, k):  # in_dim=cnn_out_dim
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.zeros(in_dim, k))
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), output shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.t() + 0.5 * pair_interactions
        return output.view(-1, 1)  # output shape(batch_size, 1)


class SourceNet(nn.Module):

    def __init__(self, config, word_emb):
        super(SourceNet, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(config, word_dim=self.embedding.embedding_dim, review_count=config.review_count)
        self.cnn_i = CNN(config, word_dim=self.embedding.embedding_dim, review_count=config.review_count)
        self.transform = nn.Sequential(
            nn.Linear(config.cnn_out_dim * 2, config.cnn_out_dim),
            nn.Tanh(),
            nn.Linear(config.cnn_out_dim, config.cnn_out_dim),
            nn.Tanh(),
            nn.Dropout(p=config.dropout_prob)
        )
        self.fm = FactorizationMachine(in_dim=config.cnn_out_dim, k=8)

    def forward(self, user_reviews, item_reviews):  # input shape(batch_size, review_count, review_length)
        new_batch_size = user_reviews.shape[0] * user_reviews.shape[1]
        user_reviews = user_reviews.view(new_batch_size, -1)
        item_reviews = item_reviews.view(new_batch_size, -1)

        u_vec = self.embedding(user_reviews)
        i_vec = self.embedding(item_reviews)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        trans_latent = self.transform(concat_latent)

        prediction = self.fm(trans_latent.detach())  # Detach forward
        return trans_latent, prediction

    def trans_param(self):
        return [x for x in self.cnn_u.parameters()] + \
               [x for x in self.cnn_i.parameters()] + \
               [x for x in self.transform.parameters()]


class TargetNet(nn.Module):

    def __init__(self, config, word_emb):
        super(TargetNet, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn = CNN(config, word_dim=self.embedding.embedding_dim, review_count=1)
        self.fm = nn.Sequential(
            nn.Dropout(config.dropout_prob),  # Since cnn did not dropout, dropout before FM.
            FactorizationMachine(in_dim=config.cnn_out_dim, k=8)
        )

    def forward(self, reviews):  # input shape(batch_size, review_length)
        vec = self.embedding(reviews)
        cnn_latent = self.cnn(vec)
        prediction = self.fm(cnn_latent)
        return cnn_latent, prediction
