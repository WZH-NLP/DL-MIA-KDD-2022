import pandas as pd
filename = 'ml-1m_Smember_train'
data = pd.read_csv(filename, sep=',', names=['SessionID', 'ItemID', 'Time'], skiprows=1)

Popularity = data.groupby('SessionID').size()
print(Popularity)
popular_item = Popularity.values.tolist()
print(popular_item)
