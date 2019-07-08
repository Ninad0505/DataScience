import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

path = 'C:\Academics\Python\Data_Science\Automobile\Automobile_data.csv'
df = pd.read_csv(path)
# print(df.head())
print(df.dtypes)
# print(df.describe())

df.dropna(subset = ['price'], axis = 0, inplace = True)
df["normalized-losses"].replace( to_replace="?", value=0)
# #
df.replace(to_replace="?", value = 0, inplace= True)
#
df['normalized-losses'] = df['normalized-losses'].astype('float')
#
# mean = df['normalized-losses'].mean()
# print(mean)

# df['length'] = (df['length'] - df['length'].mean())/ df['length'].std()
# print(df['length'])

df['price'] = df['price'].astype('float')
#
# bins = np.linspace(min(df['price']), max(df['price']), 4)
# group_names = ['low', 'medium', 'high']
# df['price-binned'] = pd.cut(df['price'], bins, labels= group_names, include_lowest=True)
#
# print(df['price-binned'])

# drive_wheels_counts = df["drive-wheels"].value_counts()
# drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'})
# drive_wheels_counts.index.name = 'drive-wheels'
# print(drive_wheels_counts)

# sns.boxplot( x = "drive-wheels", y = "price", data = df)


# x = df["engine-size"]
# y = df["price"]
# plt.scatter(x,y)
#
# plt.title("scatter plot")
# plt.ylabel("price")
# plt.xlabel("Engine-Sizw")
# plt.show()



# df_test = df[['drive-wheels', 'body-style', 'price']]
# df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()
# print(df_grp)
#
# df_pivot = df_grp.pivot( index = 'drive-wheels', columns = 'body-style')
# print(df_pivot)
#
# plt.pcolor(df_pivot, cmap = 'RdBu')
# plt.colorbar()
# plt.show()

# df_anova = df[['make', 'price']]
# grouped_anova = df_anova.groupby(['make'])
# anova_results_1 = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('subaru')['price'])
# print(anova_results_1)
#
# anova_results_2 = stats.f_oneway(grouped_anova.get_group('honda')['price'], grouped_anova.get_group('jaguar')['price'])
# print(anova_results_2)

# sns.regplot(x= 'engine-size', y= 'price', data= df)
# plt.ylim(0,)
# plt.show()
#
# sns.regplot(x= 'peak-rpm', y= 'price', data= df)
# plt.ylim(0,)
# plt.show()

# stats.pearsonr(df['horsepower'], df['price'])

