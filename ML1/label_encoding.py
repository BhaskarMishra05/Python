from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
labels = ['red', 'blue', 'green', 'blue', 'red' , 'red' , 'blue']
encoded_labels=encode.fit_transform(labels)
print(encoded_labels)