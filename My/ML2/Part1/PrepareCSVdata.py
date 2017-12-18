import pandas as pd
from sklearn.preprocessing import LabelEncoder



def load_data(dataPath):
    csv_data = pd.read_csv(dataPath)
    for col in csv_data:
        csv_data = csv_data[pd.notnull(csv_data[col])]
        if type(csv_data[col][1]) == str:
            encoder = LabelEncoder()
            housing_cat = csv_data[col]
            housing_cat = encoder.fit_transform(housing_cat)
            csv_data = csv_data.drop(col, 1)
            csv_data.insert(loc=0, column=col, value=housing_cat)
    return csv_data
#
#
# csv_path = os.path.join("datasets/housing", "housing.csv")
# csv_data = load_data(csv_path)
# print csv_data