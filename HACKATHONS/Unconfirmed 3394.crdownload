# Install Graphviz
!apt-get -qq install graphviz > /dev/null
!pip -q install graphviz

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz

# Dataset
data = {
'Previous_Fraud_History':['Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes',
                          'No','No','No','No','No','No','No','No','No','No',
                          'Yes','Yes','Yes','No','No','Yes','No','Yes','No','Yes'],
'Transaction_Amount':[90000,80000,70000,65000,60000,55000,50000,45000,40000,35000,
                      85000,75000,65000,55000,45000,35000,25000,15000,10000,5000,
                      95000,72000,38000,68000,22000,82000,18000,47000,58000,76000],
'Card_Present':['No','No','Yes','No','Yes','No','Yes','Yes','No','Yes',
                'No','Yes','No','Yes','No','Yes','No','Yes','No','Yes',
                'No','Yes','No','Yes','No','No','Yes','No','Yes','No'],
'Transaction_Location':['Foreign','Foreign','Foreign','Local','Foreign','Local','Local','Foreign','Local','Local',
                        'Foreign','Foreign','Local','Local','Foreign','Local','Foreign','Local','Local','Foreign',
                        'Foreign','Local','Foreign','Foreign','Local','Local','Foreign','Local','Foreign','Foreign'],
'Transactions_Today':[9,8,7,6,5,4,3,2,1,5,8,7,6,5,4,3,2,1,5,6,9,8,2,7,3,8,4,6,5,9],
'Class':['Fraudulent','Fraudulent','Fraudulent','Fraudulent','Fraudulent',
         'Genuine','Genuine','Fraudulent','Genuine','Genuine',
         'Fraudulent','Fraudulent','Genuine','Genuine','Fraudulent',
         'Genuine','Fraudulent','Genuine','Genuine','Fraudulent',
         'Fraudulent','Fraudulent','Genuine','Fraudulent','Genuine',
         'Fraudulent','Genuine','Fraudulent','Genuine','Fraudulent']
}

df = pd.DataFrame(data)

# Encode categorical columns
le = LabelEncoder()
for col in ['Previous_Fraud_History','Card_Present','Transaction_Location','Class']:
    df[col] = le.fit_transform(df[col])

X = df.drop('Class', axis=1)
y = df['Class']

# Train model
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

model.fit(X, y)

print("Accuracy:", accuracy_score(y, model.predict(X)))

# Display Decision Tree with True/False branches
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=X.columns,
    class_names=['Fraudulent','Genuine'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
display(graph)
