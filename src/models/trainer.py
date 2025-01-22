from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_models(X_train, y_train):
    """Train multiple classification models"""
    models = {
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models on test data"""
    eval_results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        eval_results.append({"Model": name, "Accuracy": acc})
    
    return eval_results