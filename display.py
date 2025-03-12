import joblib
import numpy as np
from sklearn.tree import _tree
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def print_tree(model, feature_names):
    """
    Function to display the decision tree rules for each tree in the model
    """
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        for i, tree_in_forest in enumerate(model.estimators_):
            print(f"\nTree {i + 1}:\n")
            tree = tree_in_forest[0]
            tree_ = tree.tree_
            _print_tree(tree_, feature_names)


def _print_tree(tree, feature_names, indent=""):
    """
    Helper function to recursively print tree rules
    """
    if tree.feature[0] != _tree.TREE_UNDEFINED:
        name = feature_names[tree.feature[0]]
        threshold = tree.threshold[0]
        print(f"{indent}if {name} <= {threshold:.2f}:")
        _print_tree(tree, feature_names, indent + "  ")
        print(f"{indent}else:  # {name} > {threshold:.2f}")
        _print_tree(tree, feature_names, indent + "  ")
    else:
        print(f"{indent}Leaf: {tree.value[0][0]:.2f}")


def display_model_formula(model_path, model_type, feature_names):
    model = joblib.load(model_path)

    if model_type == "linear_regression":
        # For Linear Regression: formula is y = mx + b (y = sum(m_i * X_i) + b)
        coef = model.coef_
        intercept = model.intercept_
        print("\nLinear Regression Model Formula:")
        formula = "y = " + " + ".join([f"{coef[i]}*{feature_names[i]}" for i in range(len(coef))])
        print(f"{formula} + {intercept}")

    elif model_type == "gradient_boosting" or model_type == "random_forest":
        # For Gradient Boosting / Random Forest, print decision trees' rules
        print(f"\n{model_type.capitalize()} Model Trees:")
        print_tree(model, feature_names)

    else:
        print("Model type not supported for formula extraction.")


# Example usage:
cities = ['berlin', 'beijing', 'moscow', 'los_angeles', 'new_york', 'sydney', 'tokyo', 'london', 'buenos_Aires',
          'mexico']
feature_names = ['YEAR']  # Adjust based on the actual features used in your models

# Display formulas for different models
model_paths = {
    "gradient_boosting_model_lr.pkl": "gradient_boosting",
    "gradient_boosting_model_svr_pol_reg.pkl": "gradient_boosting",
    "random_forest_model_pol_reg.pkl": "random_forest",
    "random_forest_model_lr.pkl": "random_forest",
    "stacked_model_svr.pkl": "linear_regression",  # Assuming stacked model contains a linear regressor
    "stacked_model_lr.pkl": "linear_regression",
    "weighted_averaging_model_polynomial.pkl": "linear_regression",  # Assuming weighted averaging uses linear regressor
    "weighted_averaging_model_lr.pkl": "linear_regression",
    "weighted_averaging_model_svr.pkl": "linear_regression"
}

for model_path, model_type in model_paths.items():
    print(f"\nModel: {model_path}")
    display_model_formula(f'models/combines/{model_path}', model_type, feature_names)
