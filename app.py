# In this program we will apply various ML algorithms to the built in datasets in scikit-learn and some datasets from kaggle

# Importing required Libraries
# Importing Numpy
import numpy as np

# To read csv file
import pandas as pd

# Importing datasets from sklearn
from sklearn import datasets

# For splitting between training and testing
from sklearn.model_selection import train_test_split

# Importing Algorithm for Simple Vector Machine
# from sklearn.svm import SVC, SVR # Removed as per request

# Importing Knn algorithm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Importing  Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Importing Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Importing Naive Bayes algorithm
# from sklearn.naive_bayes import GaussianNB # Removed as per request

# Importing Linear and Logistic Regression
# from sklearn.linear_model import LinearRegression # Removed as per request
from sklearn.linear_model import LogisticRegression


# Importing accuracy score and mean_squared_error
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error

# Importing PCA for dimension reduction
from sklearn.decomposition import PCA

# For Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# For model deployment
import streamlit as st

# Importing Label Encoder
# For converting string to int
from sklearn.preprocessing import LabelEncoder

# To Disable Warnings
# st.set_option("deprecation.showPyplotGlobalUse", False) # This is fine commented out or removed
import warnings

warnings.filterwarnings("ignore")


# Now we need to load the builtin dataset
# For the other dataset we will read the csv file from the dataset folder
# This is done using the load_dataset_name function
def load_dataset(Data):

    if Data == "Iris":
        return datasets.load_iris()
    elif Data == "Wine":
        return datasets.load_wine()
    elif Data == "Breast Cancer":
        return datasets.load_breast_cancer()
    elif Data == "Diabetes":
        return datasets.load_diabetes()
    elif Data == "Digits":
        return datasets.load_digits()
    elif Data == "Salary":
        return pd.read_csv("Dataset/Salary_dataset.csv")
    elif Data == "Naive Bayes Classification": # Dataset name, not algorithm
        return pd.read_csv("Dataset/Naive-Bayes-Classification-Data.csv")
    elif Data == "Heart Disease Classification":
        return pd.read_csv("Dataset/Updated_heart_prediction.csv")
    elif Data == "Titanic":
        return pd.read_csv("Dataset/Preprocessed Titanic Dataset.csv")
    else:
        return pd.read_csv("Dataset/car_evaluation.csv")


# Now after this we need to split between input and output
# Defining Function for Input and Output
def Input_output(data, data_name):

    if data_name == "Salary":
        X, Y = data["YearsExperience"].to_numpy().reshape(-1, 1), data[
            "Salary"
        ].to_numpy().reshape(-1, 1)

    elif data_name == "Naive Bayes Classification":
        X, Y = data.drop("diabetes", axis=1), data["diabetes"]

    elif data_name == "Heart Disease Classification":
        X, Y = data.drop("output", axis=1), data["output"]

    elif data_name == "Titanic":
        X, Y = (
            data.drop(
                columns=["survived", "home.dest", "last_name", "first_name", "title"],
                axis=1,
            ),
            data["survived"],
        )

    elif data_name == "Car Evaluation":

        df = data.copy() # Use a copy to avoid modifying the original DataFrame in memory if re-used

        # For converting string columns to numeric values
        le = LabelEncoder()

        # Function to convert string values to numeric values
        # Apply LabelEncoder to each column
        for col in df.columns:
            if df[col].dtype == 'object': # Check if column is of object type (likely string)
                df[col] = le.fit_transform(df[col])
        
        X, Y = df.drop(["unacc"], axis=1), df["unacc"]

    else:
        # We use data.data as we need to copy data to X which is Input
        X = data.data
        # Since this is built in dataset we can directly load output or target class by using data.target function
        Y = data.target

    return X, Y


# Adding Parameters so that we can select from various parameters for classifier
def add_parameter_classifier_general(algorithm):

    # Declaring a dictionary for storing parameters
    params = dict()

    # Deciding parameters based on algorithm
    # SVM removed
    # if algorithm == "SVM":
    #     c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
    #     kernel_custom = st.sidebar.selectbox(
    #         "Kernel", ("linear", "poly ", "rbf", "sigmoid") # "poly " has a trailing space, should be "poly"
    #     )
    #     params["C"] = c_regular
    #     params["kernel"] = kernel_custom

    # Adding Parameters for KNN
    if algorithm == "KNN": # Changed from elif to if as SVM is removed

        # Adding number of Neighbour in Classifier
        k_n = st.sidebar.slider("Number of Neighbors (K)", 1, 20, key="k_n_slider_classifier") # Added key for uniqueness
        # Adding in dictionary
        params["K"] = k_n
        # Adding weights
        weights_custom = st.sidebar.selectbox("Weights", ("uniform", "distance"))
        # Adding to dictionary
        params["weights"] = weights_custom

    # Naive Bayes removed
    # elif algorithm == "Naive Bayes":
    #     st.sidebar.info(
    #         "This is a simple Algorithm. It doesn't have Parameters for Hyper-tuning."
    #     )

    # Adding Parameters for Decision Tree
    elif algorithm == "Decision Tree":

        # Taking max_depth
        max_depth = st.sidebar.slider("Max Depth", 2, 17, key="dt_max_depth_classifier")
        # Adding criterion
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"), key="dt_criterion_classifier")
        # Adding splitter
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"), key="dt_splitter_classifier")
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter

        try:
            random = st.sidebar.text_input("Enter Random State", key="dt_random_classifier")
            if random: # Check if input is not empty
                 params["random_state"] = int(random)
            else:
                 params["random_state"] = None # Or a default like 42
        except ValueError:
            st.sidebar.warning("Please enter a valid integer for Random State.")
            params["random_state"] = 42 # Default on error

    # Adding Parameters for Random Forest
    elif algorithm == "Random Forest":

        max_depth = st.sidebar.slider("Max Depth", 2, 17, key="rf_max_depth_classifier")
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, key="rf_n_estimators_classifier") # Increased max to 100
        criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy", "log_loss"), key="rf_criterion_classifier")
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        
        try:
            random = st.sidebar.text_input("Enter Random State", key="rf_random_classifier")
            if random:
                 params["random_state"] = int(random)
            else:
                 params["random_state"] = None # Or a default like 42
        except ValueError:
            st.sidebar.warning("Please enter a valid integer for Random State.")
            params["random_state"] = 42


    # Adding Parameters for Logistic Regression (else block)
    elif algorithm == "Logistic Regression": # Explicitly stated for clarity
        c_regular = st.sidebar.slider("C (Regularization)", 0.01, 10.0, key="logreg_c")
        params["C"] = c_regular
        # fit_intercept is True by default and usually kept True. Not a common hyperparameter to tune for users.
        # params["fit_intercept"] = st.sidebar.selectbox("Fit Intercept", (True, False), key="logreg_fit_intercept")
        penalty = st.sidebar.selectbox("Penalty", ("l2", None), key="logreg_penalty") # 'l1' requires solver 'liblinear' or 'saga'
        params["penalty"] = penalty
        # n_jobs is for parallelization, usually -1 or 1.
        # params["n_jobs"] = st.sidebar.selectbox("Number of Jobs", (None, -1, 1), key="logreg_n_jobs")


    return params


# Adding Parameters so that we can select from various parameters for regressor
def add_parameter_regressor(algorithm):
    params = dict()
    if algorithm == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 17, key="dt_max_depth_regressor")
        criterion = st.sidebar.selectbox(
            "Criterion", ("squared_error", "friedman_mse", "absolute_error", "poisson"), key="dt_criterion_regressor" # squared_error is common
        )
        splitter = st.sidebar.selectbox("Splitter", ("best", "random"), key="dt_splitter_regressor")
        params["max_depth"] = max_depth
        params["criterion"] = criterion
        params["splitter"] = splitter
        try:
            random = st.sidebar.text_input("Enter Random State", key="dt_random_regressor")
            if random:
                 params["random_state"] = int(random)
            else:
                 params["random_state"] = None # Or a default like 42
        except ValueError:
            st.sidebar.warning("Please enter a valid integer for Random State.")
            params["random_state"] = 42
            
    # Linear Regression removed
    # elif algorithm == "Linear Regression":
    #     fit_intercept = st.sidebar.selectbox("Fit Intercept", (True, False), key="linreg_fit_intercept") # Boolean directly
    #     params["fit_intercept"] = fit_intercept
    #     # n_jobs = st.sidebar.selectbox("Number of Jobs", (None, -1, 1), key="linreg_n_jobs")
    #     # params["n_jobs"] = n_jobs

    elif algorithm == "Random Forest": # Changed from else to elif
        max_depth = st.sidebar.slider("Max Depth", 2, 17, key="rf_max_depth_regressor")
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100, key="rf_n_estimators_regressor") # Increased max
        criterion = st.sidebar.selectbox(
            "Criterion", ("squared_error", "absolute_error", "friedman_mse", "poisson"), key="rf_criterion_regressor"
        )
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        try:
            random = st.sidebar.text_input("Enter Random State", key="rf_random_regressor")
            if random:
                 params["random_state"] = int(random)
            else:
                 params["random_state"] = None
        except ValueError:
            st.sidebar.warning("Please enter a valid integer for Random State.")
            params["random_state"] = 42
    # KNN Regressor parameters (K, weights) are handled by add_parameter_classifier_general
    # SVR parameters (C, kernel) were handled by add_parameter_classifier_general, but SVM/SVR removed.
    return params


# Now we will build ML Model for this dataset and calculate accuracy for that for classifier
def model_classifier(algorithm, params):
    if algorithm == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"], weights=params["weights"])
    # SVM removed
    # elif algorithm == "SVM":
    #     return SVC(C=params["C"], kernel=params["kernel"].strip()) # Ensure kernel name is clean
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth"), # Use .get for safety if param missing
            criterion=params["criterion"],
            splitter=params["splitter"],
            random_state=params.get("random_state"),
        )
    # Naive Bayes removed
    # elif algorithm == "Naive Bayes":
    #     return GaussianNB()
    elif algorithm == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth"),
            criterion=params["criterion"],
            random_state=params.get("random_state"),
        )
    # Linear Regression (incorrectly placed here) removed
    # elif algorithm == "Linear Regression":
    #     # This was incorrect, Linear Regression is a regressor
    #     # Forcing an error or returning None would be better if it was selectable as classifier
    #     st.error("Linear Regression cannot be used as a classifier.")
    #     return None 
    elif algorithm == "Logistic Regression": # Changed from else to elif
        return LogisticRegression(
            C=params.get("C", 1.0), # Default C is 1.0
            penalty=params.get("penalty", "l2"), # Default penalty is 'l2'
            random_state=params.get("random_state"), # LogisticRegression also has random_state
            # fit_intercept=params.get("fit_intercept", True),
            # n_jobs=params.get("n_jobs")
            solver='liblinear' # Good default, supports L1/L2
        )
    else:
        st.error(f"Unknown classifier: {algorithm}")
        return None


# Now we will build ML Model for this dataset and calculate accuracy for that for regressor
def model_regressor(algorithm, params):
    if algorithm == "KNN":
        # K and weights come from add_parameter_classifier_general
        return KNeighborsRegressor(n_neighbors=params["K"], weights=params["weights"])
    # SVR (tied to SVM) removed
    # elif algorithm == "SVM":
    #     # C and kernel come from add_parameter_classifier_general
    #     return SVR(C=params["C"], kernel=params["kernel"].strip())
    elif algorithm == "Decision Tree":
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth"),
            criterion=params["criterion"],
            splitter=params["splitter"],
            random_state=params.get("random_state"),
        )
    elif algorithm == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params.get("max_depth"),
            criterion=params["criterion"],
            random_state=params.get("random_state"),
        )
    # Linear Regression (previously else block) removed
    # else: # Was Linear Regression
    #     return LinearRegression(
    #         fit_intercept=params.get("fit_intercept", True),
    #         # n_jobs=params.get("n_jobs")
    #     )
    else:
        st.error(f"Unknown regressor: {algorithm}")
        return None


# Now we will write the dataset information
def info(data_name, algorithm, algorithm_type, data, X, Y):
    st.header(f"Dataset: {data_name}")
    st.subheader(f"Algorithm: {algorithm} ({algorithm_type})")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape of Input Features (X): ", X.shape)
    with col2:
        st.write("Number of Unique Target Values: ", len(np.unique(Y)))

    if data_name not in ["Diabetes", "Salary"]: # These are primarily regression or simple structure
        # Prepare target names
        target_names_map = {
            "Iris": data.target_names,
            "Wine": data.target_names,
            "Breast Cancer": data.target_names,
            "Digits": data.target_names,
            "Naive Bayes Classification": ["Not Diabetic", "Diabetic"], # Custom for this dataset
            "Heart Disease Classification": ["Less Chance Of Heart Attack", "High Chance Of Heart Attack"],
            "Titanic": ["Not Survived", "Survived"],
            "Car Evaluation": ["Unacceptable", "Acceptable", "Good", "Very Good"] # Based on typical Car Eval labels
        }
        
        if data_name in target_names_map:
            unique_targets = sorted(np.unique(Y))
            current_target_names = target_names_map[data_name]
            
            # Ensure we don't go out of bounds if unique_targets has more values than names provided
            # (e.g. for Car Evaluation, target values are 0,1,2,3)
            display_target_names = [current_target_names[i] if i < len(current_target_names) else f"Class {i}" for i in unique_targets]

            df_classes = pd.DataFrame({
                "Target Value (Encoded)": unique_targets,
                "Target Name": display_target_names
            })
            st.write("Target Classes Information:")
            st.markdown(df_classes.to_markdown(index=False), unsafe_allow_html=True)
    st.write("---")


def choice_classifier(data, data_name, X_plot, Y, fig_ax): # X_plot is PCA-transformed
    ax = fig_ax
    if data_name == "Diabetes": # Even if used for classification, it's numeric features
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=Y, cmap="viridis", alpha=0.8, edgecolor='k')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
    elif data_name == "Digits":
        # For digits, using a qualitative colormap might be better if many classes
        scatter = sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=Y, palette="deep", alpha=0.6, ax=ax, legend='full')
    elif data_name == "Salary": # This is typically a regression dataset
        # If used for classification (e.g. binned salary), adapt. Original code used actual values.
        # For PCA plot, X_plot would be 1D if only 'YearsExperience' used and PCA(1)
        if X_plot.shape[1] > 1:
             sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=Y, palette="viridis", data=data, ax=ax)
        else: # If X_plot is 1D after PCA (e.g. from single feature)
             sns.scatterplot(x=X_plot[:, 0], y=Y, hue=Y, palette="viridis", data=data, ax=ax) # Plot against target Y
        ax.set_xlabel("Principal Component 1" if X_plot.shape[1] > 1 else "YearsExperience (or PC1)")
        ax.set_ylabel("Principal Component 2" if X_plot.shape[1] > 1 else "Salary (Target)")

    elif data_name == "Naive Bayes Classification": # Dataset name
        sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=Y, palette="coolwarm", alpha=0.7, ax=ax)
    else: # General case for other classification datasets
        sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=Y, palette="Set1", alpha=0.7, ax=ax)

    ax.set_title(f"PCA Plot of {data_name} Dataset (Classification View)")
    if data_name != "Salary": # Salary specific labels handled above
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
    return ax.figure


def choice_regressor(X_orig, x_test_orig, predict, data, data_name, Y_orig, fig_ax):
    ax = fig_ax
    # For regression, PCA might not be the best for visualizing original feature vs prediction.
    # Plotting one original feature vs target, and the prediction line.
    # We'll use the first feature of X_orig for simplicity, or specific features for known datasets.

    if data_name == "Salary":
        # X_orig is YearsExperience.to_numpy().reshape(-1, 1)
        # x_test_orig is a subset of YearsExperience
        ax.scatter(X_orig, Y_orig, color='skyblue', label='Actual Salary', alpha=0.7, edgecolor='k')
        # Sort x_test for a clean line plot if it's not sorted
        sort_axis = x_test_orig[:,0].argsort()
        ax.plot(x_test_orig[:,0][sort_axis], predict[sort_axis], color='red', linewidth=2, label='Predicted Salary (Best Fit Line)')
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.set_title("Salary Prediction")
    elif data_name == "Diabetes":
        # Plotting first feature of diabetes dataset vs target
        ax.scatter(X_orig[:, 0], Y_orig, color='lightcoral', label='Actual Value', alpha=0.6, edgecolor='k')
        # For plotting prediction line, it's tricky if x_test has multiple features.
        # We'd typically plot against one feature, holding others constant or averaging.
        # Or, plot predicted vs actual y_values.
        ax.scatter(x_test_orig[:,0], predict, color='darkred', marker='x', label='Predicted Value (on test set)', s=50)
        # A true "best fit line" across multi-D space is hard to show on 2D scatter of one feature.
        # Instead, let's plot y_test vs y_pred
        # plt.figure() # Create a new figure for y_test vs y_pred
        # plt.scatter(y_test, predict, alpha=0.7)
        # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', lw=2, color='red') # Diagonal line
        # plt.xlabel("Actual Values (y_test)")
        # plt.ylabel("Predicted Values")
        # plt.title("Actual vs. Predicted Values")
        # st.pyplot(plt)

        ax.set_xlabel("Feature 1 (e.g., BMI or Age)") # Example, assuming 1st feature
        ax.set_ylabel("Diabetes Progression (Target)")
        ax.set_title(f"{data_name} Regression (Feature 1 vs Target)")

    else: # Generic regression plot: Actual vs Predicted
        # Create a temporary new figure for this specific plot type if needed, or ensure ax is clear.
        # It's better to plot y_test vs predict for general regression tasks.
        # This requires y_test to be passed to this function.
        # The current implementation tries to plot on the PCA scatter, which is not ideal for regression line.
        # For now, let's just show the original data points (PCA-ed if X_plot is used).
        # X_plot = pca_plot(data_name, X_orig, Y_orig) # PCA transformation
        # ax.scatter(X_plot[:,0], Y_orig, c='blue', label='Actual Data (PC1 vs Target)', alpha=0.5)
        # This part is complex as x_test_orig might be multi-dimensional.
        # A simple line plot is only meaningful if x_test_orig is 1D or sorted on one dimension.
        # Fallback to a simple scatter of original data if nothing specific.
        # This function needs y_test to plot actual vs predicted effectively.
        # For now, let's assume we are plotting one of the original features from x_test_orig vs predictions
        ax.scatter(x_test_orig[:,0], predict, color='green', label='Predictions against 1st feature of X_test', marker='x')
        ax.set_xlabel("First Feature of X_test")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"{data_name} Regression Predictions")


    ax.legend()
    return ax.figure


def data_model_description(algorithm, algorithm_type, data_name, data, X, Y):
    info(data_name, algorithm, algorithm_type, data, X, Y)

    # Parameter selection
    if (algorithm_type == "Regressor") and (
        algorithm == "Decision Tree" or algorithm == "Random Forest"
    ):
        params = add_parameter_regressor(algorithm)
    else: # For all classifiers, and for KNN Regressor (uses KNN classifier params)
        params = add_parameter_classifier_general(algorithm)

    # Model building
    if algorithm_type == "Regressor":
        model = model_regressor(algorithm, params)
    else:
        model = model_classifier(algorithm, params)

    if model is None:
        st.error("Model could not be created. Please check selections.")
        return

    # Data splitting
    # Ensure Y is 1D for train_test_split if it's a column vector
    if Y.ndim > 1 and Y.shape[1] == 1:
        Y_flat = Y.ravel()
    else:
        Y_flat = Y
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y_flat, test_size=0.2, random_state=42)

    # Training
    try:
        model.fit(x_train, y_train)
    except Exception as e:
        st.error(f"Error during model training: {e}")
        return

    # Prediction
    predict = model.predict(x_test)

    # Results and Plotting
    st.subheader("Model Performance")
    if algorithm_type == "Classifier":
        train_acc = model.score(x_train, y_train)
        test_acc = accuracy_score(y_test, predict)
        st.write(f"Training Accuracy: {train_acc*100:.2f}%")
        st.write(f"Testing Accuracy: {test_acc*100:.2f}%")
    else: # Regressor
        mse = mean_squared_error(y_test, predict)
        mae = mean_absolute_error(y_test, predict)
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
        train_score = model.score(x_train, y_train) # R^2 score for regressors
        test_score = model.score(x_test, y_test)   # R^2 score for regressors
        st.write(f"Training R² Score: {train_score:.4f}")
        st.write(f"Test R² Score: {test_score:.4f}")

    st.subheader("Data Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))

    # PCA for visualization (mostly for classification)
    if X.shape[1] > 2 and data_name != "Salary": # Salary is 1D input, PCA not standard for it
        X_pca = pca_plot_transform(data_name, X, Y_flat) # Pass Y_flat for consistency
    elif data_name == "Salary":
        X_pca = X # Use original X for salary (YearsExperience)
    else: # If X has 1 or 2 features already
        X_pca = X

    if algorithm_type == "Classifier":
        # For classifiers, plot PCA of all X data, colored by true Y
        choice_classifier(data, data_name, X_pca, Y_flat, ax)
    else: # Regressor
        # For regressors, plot original feature vs target, and predictions.
        # choice_regressor needs original x_test, not PCA'd x_test generally.
        choice_regressor(X, x_test, predict, data, data_name, Y_flat, ax) # Pass original X and Y_flat

    st.pyplot(fig)


# Doing PCA(Principal Component Analysis) on the dataset and then plotting it
def pca_plot_transform(data_name, X, Y): # Y is passed for context if needed, not used in PCA fit
    # Plotting Dataset
    # Since there are many dimensions, first we will do Principle Component analysis to do dimension reduction and then plot
    pca = PCA(n_components=2)
    
    # Salary is usually 1D feature, PCA might not be standard. Naive Bayes dataset also might be low-dim.
    # The original code applied PCA to almost everything except Salary.
    if data_name != "Salary": # And X has more than 2 features
        if X.shape[1] > 2:
            X_transformed = pca.fit_transform(X)
            return X_transformed
        else: # Already 2 or fewer features
            return X 
    return X # Return original X if no PCA applied


# Main Function
def main():
    st.sidebar.header("Model Configuration")
    data_name = st.sidebar.selectbox(
        "Select Dataset",
        (
            "Iris",
            "Breast Cancer",
            "Wine",
            "Diabetes", # Regression dataset, can be used for classification too
            "Digits",
            "Salary", # Simple regression
            "Naive Bayes Classification", # This is a dataset name
            "Car Evaluation",
            "Heart Disease Classification",
            "Titanic",
        ),
    )

    algorithm = st.sidebar.selectbox(
        "Select Supervised Learning Algorithm",
        (
            "KNN",
            "Decision Tree",
            "Random Forest",
            "Logistic Regression",
            # SVM, Naive Bayes, Linear Regression removed
        ),
    )
    
    # Determine if algorithm type selection is needed
    if algorithm == "Logistic Regression":
        st.sidebar.info(f"{algorithm} is a Classification algorithm.")
        algorithm_type = "Classifier"
    # elif algorithm == "Linear Regression": # Removed
    #     st.sidebar.info(f"{algorithm} is a Regression algorithm.")
    #     algorithm_type = "Regressor"
    # elif algorithm == "Naive Bayes": # Removed
    #     st.sidebar.info(f"{algorithm} is a Classification algorithm.")
    #     algorithm_type = "Classifier"
    else: # KNN, Decision Tree, Random Forest can be both
        algorithm_type = st.sidebar.selectbox(
            "Select Algorithm Type", ("Classifier", "Regressor")
        )

    data = load_dataset(data_name)
    X, Y = Input_output(data, data_name)

    if X is not None and Y is not None:
        data_model_description(algorithm, algorithm_type, data_name, data, X, Y)
    else:
        st.error("Failed to load data or define X, Y.")


# Starting Execution of the Program
if __name__ == "__main__":
    st.set_page_config(page_title="HyperTuneML Platform", layout="wide")
    
  
    # st.title("HyperTuneML Platform")
    st.markdown("### Explore Machine Learning Algorithms on Various Datasets")
    st.markdown("""
    Select a dataset and a supervised learning algorithm from the sidebar. 
    Adjust hyperparameters to see how they affect model performance and visualizations.
    """)

    main()