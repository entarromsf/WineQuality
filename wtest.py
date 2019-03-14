import pandas
import matplotlib.pyplot as plt
import numpy as np





train_data = pandas.read_csv("./WineQualityTrain.csv", encoding='utf-8', low_memory=False, na_values='\\N').fillna(0)
shuffled_rows = np.random.permutation(train_data.index)

train_data = train_data.iloc[shuffled_rows]
train_data.hist()
plt.show()