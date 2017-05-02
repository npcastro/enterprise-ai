import mxnet
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data and basic statistics
hr_data = pd.read_csv("data/hr.csv")
hr_data.head()
hr_data.info()
hr_data.describe()

# Convert some variables to factors
hr_data[['Work_accident', 'promotion_last_5years']] = \
    hr_data[['Work_accident', 'promotion_last_5years']].astype(object)

sample_size = int(0.7 * hr_data.shape[0])
sample_seed = 0

X = hr_data.drop('left', axis=1)
y = hr_data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=sample_size, random_state=sample_seed)

