import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

close_prices = pd.read_csv('close_prices.csv')
close_prices.drop('date', axis=1, inplace=True)

pca = PCA(n_components=10)
pca.fit(close_prices)
asd = pca.transform(close_prices)
print close_prices.columns[np.argmax(pca.components_[0])]

djia_index = pd.read_csv('djia_index.csv')
result = np.corrcoef(x=djia_index['^DJI'], y=asd[:, 0])
