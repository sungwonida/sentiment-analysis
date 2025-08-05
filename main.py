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
