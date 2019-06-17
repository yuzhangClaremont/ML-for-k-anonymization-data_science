
# Code example
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


# def prepare_country_stats(oecd_bli, gdp_per_capita):
#     oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
#     oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
#     gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
#     gdp_per_capita.set_index("Country", inplace=True)
#     full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
#                                   left_index=True, right_index=True)
#     full_country_stats.sort_values(by="GDP per capita", inplace=True)
#     remove_indices = [0, 1, 6, 8, 33, 34, 35]
#     keep_indices = list(set(range(36)) - set(remove_indices))
#     return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

datapath = "data/"
# Load the data
# oecd_bli = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',') # Basicl_Stats.csv
basic = pd.read_csv(datapath + "Basic_Stats.csv", thousands=',') # Basic_Stats.csv
valid_w = basic[basic["Weight (lbs)"].notna()]
valid_wh = valid_w[basic["Height (inches)"].notna()]
print(valid_wh)
# gdp_per_capita = pd.read_csv(datapath + "gdp_per_capita.csv",thousands=',',delimiter='\t',
#                              encoding='latin1', na_values="n/a")

# # Prepare the data
# country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[valid_wh["Weight (lbs)"]]
Y = np.c_[valid_wh["Height (inches)"]]
# # x1 = [country_stats["GDP per capita"]]
# # y1 = [country_stats["Life satisfaction"]]


# # print(x1)

# # Visualize the data
valid_wh.plot(kind='scatter', x="Weight (lbs)", y='Height (inches)')
plt.show()

# # Select a linear model
model = linear_model.LinearRegression()

# # Train the model
model.fit(X, Y)
# # model.fit(x1, y1)

# # Make a prediction for Cyprus
X_new = [[178]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]