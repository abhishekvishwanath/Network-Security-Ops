import sys
from networksecurity.entity.artifacts_entity import ClassificationMetricArtifact
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from networksecurity.constants import MODEL_TRAINER_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

def get_classification_metrics(y_true, y_pred)->ClassificationMetricArtifact:
    try:
        logging.info("Calculating classification metrics")
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logging.info(f"Metrics calculated - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return ClassificationMetricArtifact(precision=precision, recall=recall, f1_score=f1)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        logging.info("Starting model evaluation with hyperparameter tuning")
        report = {}
        trained_models = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            
            # Get parameters for current model
            model_params = params.get(model_name, {})
            
            if model_params:
                # Perform GridSearchCV if parameters are provided
                logging.info(f"Performing hyperparameter tuning for {model_name}")
                gs = GridSearchCV(model, model_params, cv=3, scoring='f1', n_jobs=-1)
                gs.fit(x_train, y_train)
                
                # Use best model
                best_model = gs.best_estimator_
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                # Train with default parameters
                best_model = model
                best_model.fit(x_train, y_train)
            
            # Make predictions
            y_test_pred = best_model.predict(x_test)
            
            # Calculate test score
            test_model_score = f1_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            trained_models[model_name] = best_model
            
            logging.info(f"{model_name} - Test F1 Score: {test_model_score:.4f}")
        
        logging.info("Model evaluation completed")
        return report, trained_models
        
    except Exception as e:
        raise CustomException(e, sys)


