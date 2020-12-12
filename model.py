import torch
from torch import nn


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
        latent = self.conv(vec.permute(0, 2, 1))  # output shape(new_batch_size, kernel_count, 1)
        latent = latent.view(-1, self.kernel_count * self.review_count)
        latent = self.linear(latent)
        return latent  # output shape(batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, in_dim, k):  # in_dim=cnn_out_dim
        super(FactorizationMachine, self).__init__()
        self.v = nn.Parameter(torch.full([in_dim, k], 0.001))
        self.linear = nn.Linear(in_dim, 1)
        self.linear.weight.data.normal_(mean=0, std=0.001)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), output shape(batch_size, 1)
        inter_part1 = torch.mm(x, self.v)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.t() + 0.5 * pair_interactions
        return output.view(-1, 1)  # output shape(batch_size, 1)


class SourceNet(nn.Module):

    def __init__(self, config, word_emb, extend_model=False):
        super(SourceNet, self).__init__()
        self.extend_model = extend_model
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
        for m in self.transform.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0, std=0.1).clamp_(-1, 1)
                nn.init.constant_(m.bias.data, 0.1)

        if self.extend_model:
            self.emb_u = nn.Embedding(config.user_count, config.cnn_out_dim, padding_idx=0)
            self.emb_i = nn.Embedding(config.item_count, config.cnn_out_dim, padding_idx=0)
            self.fm = FactorizationMachine(in_dim=config.cnn_out_dim * 3, k=8)
        else:
            self.fm = FactorizationMachine(in_dim=config.cnn_out_dim, k=8)

    def forward(self, user_reviews, item_reviews, user_ids, item_ids):  # shape(batch_size, review_count, review_length)
        new_batch_size = user_reviews.shape[0] * user_reviews.shape[1]
        user_reviews = user_reviews.view(new_batch_size, -1)
        item_reviews = item_reviews.view(new_batch_size, -1)

        u_vec = self.embedding(user_reviews)
        i_vec = self.embedding(item_reviews)

        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        trans_latent = self.transform(concat_latent)

        if self.extend_model:
            omega_u = self.emb_u(user_ids.view(-1))
            omega_i = self.emb_i(item_ids.view(-1))
            latent = torch.cat([omega_u, omega_i, trans_latent.detach()], dim=1)
            prediction = self.fm(latent)
        else:
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
