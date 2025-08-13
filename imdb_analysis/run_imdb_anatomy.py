import imdb_anatomy

# 0 Get the dataset
df = imdb_anatomy.get_imdb_dataset_as_df("train")

# 1 Review-length distribution
imdb_anatomy.length_distribution(df)

# 2 Class-balanced length comparison
imdb_anatomy.class_balanced_length(df)

# 3 Quick peek at extreme outliers
imdb_anatomy.extreme_outliers(df)

# 4 Top unigrams & bigrams per class
imdb_anatomy.top_uni_bigrams(df)

# 5 Word-cloud (fun but also useful)
imdb_anatomy.word_cloud(df)

# 6 Vocabulary size & OOV rate (after tokeniser)
imdb_anatomy.vocab_size_oov_rate(df)

# # 7 Sentiment vs. review length correlation
# imdb_anatomy.sentiment_review_length(df)

# # 8 Stratified train/valid split (for your own experiments)
# from sklearn.model_selection import train_test_split
# train_df, valid_df = train_test_split(
#     df, test_size=0.1, stratify=df["label"], random_state=42)
# print(train_df["label"].value_counts(normalize=True))

# # 9 Export to PyTorch Dataset quickly
# import torch
# from torch.utils.data import Dataset

# class IMDBDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len=256):
#         self.df = dataframe.reset_index(drop=True)
#         self.tok = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         text  = self.df.loc[idx, "text"]
#         label = int(self.df.loc[idx, "label"])
#         enc   = self.tok(text,
#                          max_length=self.max_len,
#                          padding="max_length",
#                          truncation=True,
#                          return_tensors="pt")
#         item  = {k: v.squeeze(0) for k, v in enc.items()}
#         item["labels"] = torch.tensor(label)
#         return item

# dataset = IMDBDataset(train_df, tok)
