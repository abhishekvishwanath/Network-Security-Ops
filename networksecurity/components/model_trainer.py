import os
import sys
import numpy as np
import mlflow
import dagshub
dagshub.init(repo_owner='abhishekvishwanath', repo_name='Network-Security-Ops', mlflow=True)

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifacts_entity import ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifacts_entity import DataTransformationArtifact

from networksecurity.utils.main_utils import load_object
from networksecurity.utils.main_utils import save_object
from networksecurity.utils.main_utils import save_numpy_array_data
from networksecurity.utils.main_utils import load_numpy_array_data  
from networksecurity.utils.ml_utils import get_classification_metrics
from networksecurity.utils.ml_utils import evaluate_models

from networksecurity.constants import MODEL_TRAINER_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            # Set random seed for reproducibility
            self.random_state = 42
            np.random.seed(self.random_state)
        except Exception as e:
            raise CustomException(e, sys)
    
    def track_mlflow(self, best_model, train_model_score, x_test, y_test) -> None:
        try:
            # Set MLflow tracking URI (you might want to move this to config)
            mlflow.set_tracking_uri("http://localhost:5000")
            
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(best_model.get_params())
                
                # Log metrics
                mlflow.log_metric("f1_score", train_model_score.f1_score)
                mlflow.log_metric("precision_score", train_model_score.precision)
                mlflow.log_metric("recall_score", train_model_score.recall)
                
                # Log model with signature
                signature = infer_signature(x_test, best_model.predict(x_test))
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name="network_security_model"
                )
                
                # Log training configuration
                mlflow.log_params({
                    "model_type": best_model.__class__.__name__,
                    "feature_count": x_test.shape[1]
                })
                
                # Log the best model's score
                mlflow.log_metric("best_score", train_model_score.f1_score)
                
                logging.info(f"Successfully logged metrics and model to MLflow")
                
        except Exception as e:
            logging.warning(f"MLflow tracking failed: {str(e)}")
    
    def train_model(self, x_train, y_train, x_test, y_test) -> None:
        try:
            logging.info("Starting model training process")
            
            # Initialize models with random state for reproducibility
            models = {
                "LogisticRegression": LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,  # Increase max_iter for better convergence
                    class_weight='balanced'  # Handle class imbalance
                ),
                "KNeighborsClassifier": KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance'  # Closer neighbors have more influence
                ),
                "DecisionTreeClassifier": DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10,  # Limit tree depth
                    min_samples_split=10,  # Require more samples to split
                    class_weight='balanced'
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=10,
                    class_weight='balanced_subsample',
                    n_jobs=-1  # Use all cores
                ),
                "AdaBoostClassifier": AdaBoostClassifier(
                    random_state=self.random_state,
                    n_estimators=50
                ),
                "GradientBoostingClassifier": GradientBoostingClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1
                ),
            }
            
            params = {
                "LogisticRegression": {
                    "C": [0.1, 1, 10],
                    "solver": ["lbfgs", "sag", "saga"],
                    "penalty": ["l2"],  # Removed 'none' as it's not compatible with some solvers
                    "max_iter": [1000]
                },
                
                "KNeighborsClassifier": {
                    "n_neighbors": [3, 5, 7],
                    # "weights": ["uniform", "distance"],
                    # "metric": ["euclidean", "manhattan", "minkowski"]
                },
                
                "DecisionTreeClassifier": {
                    # "criterion": ["gini", "entropy", "log_loss"],
                    # "splitter": ["best", "random"],
                    "max_depth": [5, 10, 20],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4]
                },
                
                "RandomForestClassifier": {
                    "n_estimators": [50, 100],
                    # "criterion": ["gini", "entropy", "log_loss"],
                    "max_depth": [5, 10],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                    # "bootstrap": [True, False]
                },
                
                "AdaBoostClassifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1],
                    # "algorithm": ["SAMME", "SAMME.R"]
                },
                
                "GradientBoostingClassifier": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    # "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2, 4],
                    # "subsample": [0.5, 0.7, 1.0]
                },
                
            }

            logging.info("Evaluating multiple models with hyperparameter tuning")
            model_report, trained_models = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=trained_models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            logging.info("Calculating model performance metrics")
            y_test_pred=best_model.predict(x_test)
            test_model_score=get_classification_metrics(y_test,y_test_pred)
            y_train_pred=best_model.predict(x_train)
            train_model_score=get_classification_metrics(y_train,y_train_pred)
            logging.info(f"Train F1 Score: {train_model_score.f1_score}, Test F1 Score: {test_model_score.f1_score}")
            save_object("final_model/model.pkl",best_model)

# Track experiments with mlflow
            try:
                self.track_mlflow(best_model, train_model_score, x_test, y_test)
            except Exception as e:
                logging.warning(f"MLflow tracking failed but continuing: {str(e)}")
            
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_model_score,
                test_metric_artifact=test_model_score
            )

            return model_trainer_artifact


        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading training and test data")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, 0:-1], train_arr[:, -1],
                test_arr[:, 0:-1], test_arr[:, -1]
            )

            model_trainer_artifact=self.train_model(x_train, y_train,x_test,y_test)
            return model_trainer_artifact


            
        except Exception as e:
            raise CustomException(e, sys)
