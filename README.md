Download Link: https://assignmentchef.com/product/solved-cse5334-assignment-2-multiclassclassification
<br>
Part of the materials in this instruction notebook are adapted from “Introduction to Machine Learning with Python” by Andreas C. Mueller and Sarah Guido.

To run the examples in this notebook and to finish your assignment, you need a few Python modules. If you already have a Python installation set up, you can use pip to install all of these packages:

$ pip install numpy matplotlib ipython jupyter scikit-learn pandas graphviz

In your python code, you will always need to import a subset of the following modules.

In [3]: <strong>import</strong> <strong>numpy</strong> <strong>as</strong> <strong>np</strong> <strong>import</strong> <strong>pandas</strong> <strong>as</strong> <strong>pd</strong> <strong>import</strong> <strong>matplotlib.pyplot</strong> <strong>as</strong> <strong>plt</strong> <strong>import</strong> <strong>graphviz</strong>




<strong>from</strong> <strong>IPython.display</strong> <strong>import</strong> display <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> cross_val_score <strong>from</strong> <strong>sklearn.neighbors</strong> <strong>import</strong> KNeighborsClassifier <strong>from</strong> <strong>sklearn.svm</strong> <strong>import</strong> LinearSVC <strong>from</strong> <strong>sklearn.naive_bayes</strong> <strong>import</strong> GaussianNB <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> export_graphviz

<h1>1.  The Breast Cancer Dataset</h1>

The first dataset that we use in this notebook is included in scikit-learn, a popular machine learning library for Python. The dataset is the Wisconsin Breast Cancer dataset, which records clinical measurements of breast cancer tumors. Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors), and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.

The data can be loaded using the load_breast_cancer function from scikit-learn:

In [4]: cancer = load_breast_cancer() print(“cancer.keys(): <strong>{}</strong>“.format(cancer.keys()))

cancer.keys(): dict_keys([‘data’, ‘target’, ‘target_names’, ‘DESCR’, ‘f eature_names’])

Datasets that are included in scikit-learn are usually stored as Bunch objects, which contain some information about the dataset as well as the actual data. All you need to know about Bunch objects is that they behave like dictionaries, with the added benefit that you can access values using a dot (as in bunch.key instead of bunch[‘key’]).

The dataset consists of 569 data points, with 30 features each:

In [5]: print(“Shape of cancer data: <strong>{}</strong>“.format(cancer.data.shape))

Shape of cancer data: (569, 30)

Of these 569 data points, 212 are labeled as malignant and 357 as benign:

In [6]: print(“Sample counts per class:<strong>
{}</strong>“.format(

{n: v <strong>for</strong> n, v <strong>in</strong> zip(cancer.target_names, np.bincount(cancer.targ et))}))

Sample counts per class:

{‘malignant’: 212, ‘benign’: 357}

To get a description of the semantic meaning of each feature, we can have a look at the feature_names attribute:

In [7]: print(“Feature names:<strong>
{}</strong>“.format(cancer.feature_names))

Feature names:

[‘mean radius’ ‘mean texture’ ‘mean perimeter’ ‘mean area’

‘mean smoothness’ ‘mean compactness’ ‘mean concavity’

‘mean concave points’ ‘mean symmetry’ ‘mean fractal dimension’

‘radius error’ ‘texture error’ ‘perimeter error’ ‘area error’

‘smoothness error’ ‘compactness error’ ‘concavity error’

‘concave points error’ ‘symmetry error’ ‘fractal dimension error’

‘worst radius’ ‘worst texture’ ‘worst perimeter’ ‘worst area’

‘worst smoothness’ ‘worst compactness’ ‘worst concavity’

‘worst concave points’ ‘worst symmetry’ ‘worst fractal dimension’]

Let’s print out the names of the features (attributes) and the values in the target (class attribute), and the first 3 instances in the dataset.

In [8]: print(cancer.feature_names,cancer.target_names) <strong>for</strong> i <strong>in</strong> range(0,3):     print(cancer.data[i], cancer.target[i])

[‘mean radius’ ‘mean texture’ ‘mean perimeter’ ‘mean area’

‘mean smoothness’ ‘mean compactness’ ‘mean concavity’

‘mean concave points’ ‘mean symmetry’ ‘mean fractal dimension’

‘radius error’ ‘texture error’ ‘perimeter error’ ‘area error’

‘smoothness error’ ‘compactness error’ ‘concavity error’

‘concave points error’ ‘symmetry error’ ‘fractal dimension error’

‘worst radius’ ‘worst texture’ ‘worst perimeter’ ‘worst area’

‘worst smoothness’ ‘worst compactness’ ‘worst concavity’

‘worst concave points’ ‘worst symmetry’ ‘worst fractal dimension’] [‘m alignant’ ‘benign’]

[  1.79900000e+01   1.03800000e+01   1.22800000e+02   1.00100000e+03

1.18400000e-01   2.77600000e-01   3.00100000e-01   1.47100000e-01    2.41900000e-01   7.87100000e-02   1.09500000e+00   9.05300000e-01    8.58900000e+00   1.53400000e+02   6.39900000e-03   4.90400000e-02    5.37300000e-02   1.58700000e-02   3.00300000e-02   6.19300000e-03    2.53800000e+01   1.73300000e+01   1.84600000e+02   2.01900000e+03

1.62200000e-01   6.65600000e-01   7.11900000e-01   2.65400000e-01    4.60100000e-01   1.18900000e-01] 0 [  2.05700000e+01   1.77700000e+01   1.32900000e+02   1.32600000e+03

8.47400000e-02   7.86400000e-02   8.69000000e-02   7.01700000e-02    1.81200000e-01   5.66700000e-02   5.43500000e-01   7.33900000e-01    3.39800000e+00   7.40800000e+01   5.22500000e-03   1.30800000e-02    1.86000000e-02   1.34000000e-02   1.38900000e-02   3.53200000e-03    2.49900000e+01   2.34100000e+01   1.58800000e+02   1.95600000e+03

1.23800000e-01   1.86600000e-01   2.41600000e-01   1.86000000e-01    2.75000000e-01   8.90200000e-02] 0 [  1.96900000e+01   2.12500000e+01   1.30000000e+02   1.20300000e+03

1.09600000e-01   1.59900000e-01   1.97400000e-01   1.27900000e-01    2.06900000e-01   5.99900000e-02   7.45600000e-01   7.86900000e-01    4.58500000e+00   9.40300000e+01   6.15000000e-03   4.00600000e-02    3.83200000e-02   2.05800000e-02   2.25000000e-02   4.57100000e-03    2.35700000e+01   2.55300000e+01   1.52500000e+02   1.70900000e+03

1.44400000e-01   4.24500000e-01   4.50400000e-01   2.43000000e-01

3.61300000e-01   8.75800000e-02] 0

You can find out more about the data by reading cancer.DESCR if you are interested.

<h1>2.  k-Nearest Neighbor</h1>

<h2>k-Neighbors Classification</h2>

Now let’s look at how we can apply the k-nearest neighbors algorithm using scikit-learn. First, we split our data into a training and a test set so we can evaluate generalization performance:

In [9]: train_feature, test_feature, train_class, test_class = train_test_split(

cancer.data, cancer.target, stratify=cancer.target, random_state=0)

Note that this function randomly partitions the dataset into training and test sets. The randomness is controlled by a pseudo random number generator, which generates random numbers using a seed. If you fix the seed, you will actually always get the same partition (thus no randomness). That is why we set random_state=0. (We can also use any other fixed number instead of 0, to acheive the same effect.) It guarantees that you reproduce the same results in every run. It is useful in testing your programs. However, in your real production code where randomness is needed, you shouldn’t fix random_state.

Next, we instantiate the KNeighborsClassifier class. This is when we can set parameters, like the number of neighbors to use. Here, we set it to 3:

Now, we fit the classifier using the training set. For KNeighborsClassifier this means storing the dataset, so we can compute neighbors during prediction:

Out[10]: KNeighborsClassifier(algorithm=’auto’, leaf_size=30, metric=’minkowsk i’,

metric_params=None, n_jobs=1, n_neighbors=3, p=2,            weights=’uniform’)

To make predictions on the test data, we call the predict method. For each data point in the test set, this computes its nearest neighbors in the training set and finds the most common class among these:

In [11]: print(“Test set predictions:<strong>
{}</strong>“.format(knn.predict(test_feature)))

Test set predictions:

[0 0 0 1 0 1 0 0 0 1 0 0 1 1 1 1 1 0 1 0 0 0 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0

<ul>

 <li>0 1 0 0 0 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 0 1 1 0 1 1 1 1</li>

 <li>1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 0 0 1 0 1 1 1 0 0 1 1 1</li>

</ul>

0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 0]

To evaluate how well our model generalizes, we can call the score method with the test data together with the test labels:

In [12]: print(“Test set accuracy: <strong>{:.2f}</strong>“.format(knn.score(test_feature, test_cl ass)))

Test set accuracy: 0.92

We see that our model is about 92% accurate, meaning the model predicted the class correctly for 92% of the samples in the test dataset.

<h2>Analyzing KNeighborsClassifier</h2>

Let’s investigate whether we can confirm the connection between model complexity and generalization. For that, we evaluate training and test set performance with different numbers of neighbors.

The plot shows the training and test set accuracy on the y-axis against the setting of n_neighbors on the x-axis. While real-world plots are rarely very smooth, we can still recognize some of the characteristics of overfitting and underfitting. Considering a single nearest neighbor, the prediction on the training set is perfect. But when more neighbors are considered, the model becomes simpler and the training accuracy drops. The test set accuracy for using a single neighbor is lower than when using more neighbors, indicating that using the single nearest neighbor leads to a model that is too complex. On the other hand, when considering 10 neighbors, the model is too simple and performance is even worse. (It is not a typo. Yes, using less neighbors leads to more complex models. Think carefully about this.) The best performance is somewhere in the middle, using around six neighbors. Still, it is good to keep the scale of the plot in mind. The worst performance is around 88% accuracy, which might still be acceptable.

<h1>3.  Linear Support Vector Machines</h1>

Linear support vector machines (linear SVMs) is implemented in svm.LinearSVC. Let’s apply it on the brest cancer dataset.

In [20]: <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.svm</strong> <strong>import</strong> LinearSVC




cancer = load_breast_cancer()

train_feature, test_feature, train_class, test_class = train_test_split(     cancer.data, cancer.target, stratify=cancer.target, random_state=0)




linearsvm = LinearSVC(random_state=0).fit(train_feature, train_class) print(“Test set score: <strong>{:.3f}</strong>“.format(linearsvm.score(test_feature, test _class)))

Test set score: 0.825

<h1>4.  Naive Bayes Classifiers</h1>

Naive Bayes classifiers are also implemented in scikit-learn. Since the features in the breast cancer dataset are all continuous numeric attributes, let’s use GaussianNB.

In [21]: <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.naive_bayes</strong> <strong>import</strong> GaussianNB




cancer = load_breast_cancer()

train_feature, test_feature, train_class, test_class = train_test_split(     cancer.data, cancer.target, stratify=cancer.target, random_state=0)




nb = GaussianNB().fit(train_feature, train_class)

print(“Test set score: <strong>{:.3f}</strong>“.format(nb.score(test_feature, test_class )))

Test set score: 0.923

<h1>5.  Decision trees</h1>

Decision trees are also implmented in scikit-learn. Let’s use DecisionTreeClassifier.

In [22]: <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> DecisionTreeClassifier




cancer = load_breast_cancer()

train_feature, test_feature, train_class, test_class = train_test_split(     cancer.data, cancer.target, stratify=cancer.target, random_state=42)




tree = DecisionTreeClassifier(random_state=0) tree.fit(train_feature, train_class)

print(“Training set score: <strong>{:.3f}</strong>“.format(tree.score(train_feature, trai n_class)))

print(“Test set score: <strong>{:.3f}</strong>“.format(tree.score(test_feature, test_clas s)))

Training set score: 1.000

Test set score: 0.937

If we don’t restrict the depth of a decision tree, the tree can become arbitrarily deep and complex. Unpruned trees are therefore prone to overfitting and not generalizing well to new data. Now let’s apply pre-pruning to the tree, which will stop developing the tree before we perfectly fit to the training data. One option is to stop building the tree after a certain depth has been reached. In the above code, we didn’t set max_depth (i.e., max_depth= None, which is the default value). Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split instances (min_samples_split is another parameter in

DecisionTreeClassifier). Now let’s set max_depth=4, meaning only four consecutive questions can be asked. Limiting the depth of the tree decreases overfitting. This leads to a lower accuracy on the training set, but an improvement on the test set:

In [23]: <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> DecisionTreeClassifier




cancer = load_breast_cancer()

train_feature, test_feature, train_class, test_class = train_test_split(     cancer.data, cancer.target, stratify=cancer.target, random_state=42)




tree = DecisionTreeClassifier(max_depth=4, random_state=0) tree.fit(train_feature, train_class)

print(“Training set score: <strong>{:.3f}</strong>“.format(tree.score(train_feature, trai n_class)))

print(“Test set score: <strong>{:.3f}</strong>“.format(tree.score(test_feature, test_clas s)))

Training set score: 0.988

Test set score: 0.951

<h2>Analyzing Decision Trees</h2>

We can visualize the tree using the export_graphviz function from the tree module. This writes a file in the .dot file format, which is a text file format for storing graphs. We set an option to color the nodes to reflect the majority class in each node and pass the class and features names so the tree can be properly labeled:

In [24]: <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> export_graphviz export_graphviz(tree, out_file=”tree.dot”, class_names=[“malignant”, “be nign”],

feature_names=cancer.feature_names, impurity=<strong>False</strong>, fill ed=<strong>True</strong>)

<h2>Feature Importance in trees</h2>

Instead of looking at the whole tree, there are some useful properties that we can derive to summarize the workings of the tree. The most commonly used summary is feature importance, which rates how important each feature is for the decision a tree makes. It is a number between 0 and 1 for each feature, where 0 means “not used at all” and 1 means “perfectly predicts the target.” The feature importances always sum to 1:

In [26]: print(“Feature importances:<strong>
{}</strong>“.format(tree.feature_importances_))

Feature importances:

[ 0.          0.          0.          0.          0.          0.          0.

<ol>

 <li>0.          0.01019737  0.04839825  0.</li>

 <li></li>

</ol>

0.0024156   0.          0.          0.          0.          0.

0.72682851  0.0458159   0.          0.          0.0141577   0.

0.018188

0.1221132   0.01188548  0.        ]

<h1>6.  Model Evaluation</h1>

To evaluate our supervised models, so far we have split our dataset into a training set and a test set using the train_test_split function, built a model on the training set by calling the fit method, and evaluated it on the test set using the score method, which for classification computes the fraction of correctly classified samples.

<h2>Confusion Matrix</h2>

scikit-learn has its own function for producing confusion matrix. But, let’s use pandas which is a popular Python package for data analysis. Its crosstab function produces a better-looking confusion matrix.

In [27]: <strong>import</strong> <strong>pandas</strong> <strong>as</strong> <strong>pd</strong>




<strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> train_test_split <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> DecisionTreeClassifier




cancer = load_breast_cancer()

train_feature, test_feature, train_class, test_class = train_test_split(     cancer.data, cancer.target, stratify=cancer.target, random_state=42)




tree = DecisionTreeClassifier(max_depth=4, random_state=0) tree.fit(train_feature, train_class)

print(“Training set score: <strong>{:.3f}</strong>“.format(tree.score(train_feature, trai n_class)))

print(“Test set score: <strong>{:.3f}</strong>“.format(tree.score(test_feature, test_clas s)))

prediction = tree.predict(test_feature) print(“Confusion matrix:”)

print(pd.crosstab(test_class, prediction, rownames=[‘True’], colnames=[ ‘Predicted’], margins=<strong>True</strong>))

Training set score: 0.988 Test set score: 0.951

Confusion matrix:

Predicted   0   1  All

True

<ul>

 <li>49 4   53</li>

 <li>3 87   90</li>

</ul>

All        52  91  143

<h2>Cross-Validation</h2>

The reason we split our data into training and test sets is that we are interested in measuring how well our model generalizes to new, previously unseen data. We are not interested in how well our model fit the training set, but rather in how well it can make predictions for data that was not observed during training.

Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set. Cross-validation is implemented in scikit-learn using the cross_val_score function from the model_selection module. The parameters of the cross_val_score function are the model we want to evaluate, the training data, and the ground-truth labels. Let’s evaluate

DecisionTreeClassifier on the breast cancer dataset. We can control the number of folds used by setting the cv parameter. We also summarize the cross-validation accuracy by computing the mean accuracy of the multiple folds.

scikit-learn uses stratified k-fold cross-validation for classification. In stratified cross-validation, we split the data such that the proportions between classes are the same in each fold as they are in the whole dataset.

In [28]: <strong>from</strong> <strong>sklearn.model_selection</strong> <strong>import</strong> cross_val_score <strong>from</strong> <strong>sklearn.datasets</strong> <strong>import</strong> load_breast_cancer <strong>from</strong> <strong>sklearn.tree</strong> <strong>import</strong> DecisionTreeClassifier




cancer = load_breast_cancer()

tree = DecisionTreeClassifier(max_depth=4, random_state=0) scores = cross_val_score(tree, cancer.data, cancer.target, cv=5) print(“Cross-validation scores: <strong>{}</strong>“.format(scores)) print(“Average cross-validation score: <strong>{:.2f}</strong>“.format(scores.mean()))

Cross-validation scores: [ 0.92173913  0.88695652  0.9380531   0.929203

54  0.90265487]

Average cross-validation score: 0.92

<h1>7.  How to Save Your Trained Model into a Pickle Object</h1>

Python’s pickle module serializes objects so that they can be saved to a file and loaded in a program again later on. Below we show how to save your model as a pickle object and how to load