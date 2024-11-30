from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)

param_grid_svm = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm)
grid_search_svm.fit(X_train, y_train)

param_grid_tree = {'max_depth': [3, 5, 7, None]}
tree = DecisionTreeClassifier()
grid_search_tree = GridSearchCV(tree, param_grid_tree)
grid_search_tree.fit(X_train, y_train)

def get_test_sample_size(X_test):
    return len(X_test)

def get_best_kernel_svm(grid_search_svm):
    best_kernel = grid_search_svm.best_params_['kernel']
    return best_kernel

def get_decision_tree_accuracy(grid_search_tree, X_test, y_test):
    best_model_tree = grid_search_tree.best_estimator_
    y_pred_tree = best_model_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_tree)
    return accuracy

test_sample_size = get_test_sample_size(X_test)
print(f"Number of records in the test sample: {test_sample_size}")

best_kernel_svm = get_best_kernel_svm(grid_search_svm)
print(f"Best kernel for SVM: {best_kernel_svm}")

accuracy_tree = get_decision_tree_accuracy(grid_search_tree, X_test, y_test)
print(f"Accuracy of Decision Tree on test data: {accuracy_tree * 100:.2f}%")
