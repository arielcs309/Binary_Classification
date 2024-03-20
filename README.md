# Binary Classification: Machine Learning 
Using the package SKlearn.

# About the project

The project uses machine learning with the package SKlearn. The algorithm predicts weather the apple is good or bad. I used the SVC (Support Vector Classifier) and the Decision Trees. The algorithm learns how to work through the apple information which are: Size, Weight, sweetness, crunchiness, juiciness and ripeness.
At the first SVC I had to scale the X columns to train.
## Boxplot
``` bash
df.boxplot(['Size', 'Weight', 'Sweetness', 'Crunchiness',  'Juiciness', 'Ripeness']);
```
![boxplot](https://github.com/arielcs309/Binary_Classification/blob/main/Boxplot.png)

## Correlation
Price and square feet have the highest correlation. The others don't have a good correlation since they're close to zero.
![correlation](https://github.com/arielcs309/Binary_Classification/blob/main/Correlation.png)

## Support Vector Classifier
 It is a type of linear classification model in machine learning that is used for binary classification tasks. Using the SVC is like a blackbox, because you can't understand the algorithm and why it decided to say yes or no.
```bash
SEED = 20
train_x, test_x, train_y, test_y = train_test_split(scaledX, y,
                                                   random_state = SEED, test_size = 0.25,
                                                   stratify = y)
print("We'll train with %d elements and we'll test with %d elements"%(len(train_x),len(test_x)))

model = LinearSVC()
model.fit(train_x, train_y)
prediction = model.predict(test_x)

accuracy = accuracy_score(test_y, prediction)*100
print("The accuracy was %.2f%%"%accuracy)
```
## Decision Tree
We don't need to scale in a decision tree. So I will use the X that isn't Scaled
You can see below how is the algorithm working and its logic. The map shows how it predicts by Yes or No. 
The code I used to create the decision tree map:
```bash
features = X.columns

dot_data = export_graphviz(model2, out_file = None,
                           feature_names = features,
                          filled = True,
                           rounded = True,
                          class_names = ['no','yes'])
graph = graphviz.Source(dot_data)
graph
```

![DT](https://github.com/arielcs309/Binary_Classification/blob/main/Decision%20trees.png)

# Language
![python](https://github.com/arielcs309/ML-Sklearn/blob/main/python.jpg)

# Author
Ariel Sousa. 
See my [Linkedin](https://www.linkedin.com/in/ariel-candido-22684578/) and my [Kaggle](https://www.kaggle.com/arielsousa)


