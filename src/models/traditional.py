import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TraditionalMLModels:
    """传统机器学习模型集合"""
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
        }
        self.results = {}
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test) -> dict:
        """训练并评估所有模型"""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'direction_accuracy': self._direction_accuracy(y_test, y_pred)
            }
            
            print(f"\n{name} Results:")
            print(f"  MSE: {self.results[name]['mse']:.6f}")
            print(f"  MAE: {self.results[name]['mae']:.6f}")
            print(f"  R²: {self.results[name]['r2']:.4f}")
            print(f"  Direction Accuracy: {self.results[name]['direction_accuracy']:.2%}")
        
        return self.results
    
    def _direction_accuracy(self, y_true, y_pred) -> float:
        """计算方向预测准确率"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return np.mean(true_direction == pred_direction)
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """获取特征重要性"""
        model = self.results[model_name]['model']
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return df