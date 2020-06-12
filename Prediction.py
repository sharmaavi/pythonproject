import pandas as pd
import Accuracy

dfts = pd.read_csv('Dataset/test.csv')

def Test_ETL():
    X = dfts.iloc[:, :17]

    X_mar = X['marital'].replace(['single', 'divorced', 'married'], [0, -1, 1])
    X_def = X['default'].replace(['yes', 'no'], [1, 0])
    X_edu = X['education'].replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3])
    X_hou = X['housing'].replace(['yes', 'no'], [1, 0])
    X_loan = X['loan'].replace(['yes', 'no'], [1, 0])
    X_con = X['contact'].replace(['unknown', 'cellular', 'telephone'], [0, 1, 2])
    X_mon = X['month'].replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    X_job = X['job'].replace(['unknown', 'admin.', 'services', 'management', 'technician', 'retired',
                              'blue-collar', 'housemaid', 'self-employed', 'student', 'entrepreneur',
                              'unemployed'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    X_poc = X['poutcome'].replace(['unknown', 'failure', 'success', 'other'], [-1, 0, 1, 2])

    frames_test = [X['age'], X_job, X_mar, X_edu, X_def, X['balance'], X_hou, X_loan, X_con, X['day'], X_mon,
                   X['duration'], X['campaign'], X['pdays'], X['previous'], X_poc]
    X_Test = pd.concat(frames_test, axis=1)
    return X_Test


X_Test = Test_ETL()
X_Train = Accuracy.x
y_Train = Accuracy.y



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_Train)
X_Train = scaler.transform(X_Train)

scaler.fit(X_Test)
X_Test = scaler.transform(X_Test)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_Train, y_Train)
knnPred = knn.predict(X_Test)


print("Predicted Outcomes are:", knnPred)


dfts['subscribed'] = knnPred
pd.DataFrame(dfts).to_csv('Test_Results', index=None)
