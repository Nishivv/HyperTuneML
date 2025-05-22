 
# HyperTuneML Platform 🧪🔬

**Interactively explore Machine Learning algorithms on various datasets. Adjust hyperparameters and visualize their impact on model performance.**

This Gradio application allows users to select from a range of popular datasets (both built-in from scikit-learn and custom CSVs) and apply different supervised machine learning algorithms (KNN, Decision Tree, Random Forest, Logistic Regression) for either classification or regression tasks. Users can tune hyperparameters for each algorithm and immediately see the results, including performance metrics and data visualizations.

## Features

*   **Interactive UI:** Built with Gradio for a user-friendly web interface.
*   **Multiple Datasets:**
    *   Scikit-learn classics: Iris, Wine, Breast Cancer, Diabetes, Digits.
    *   Custom CSV datasets: Salary, Naive Bayes Classification (Diabetes), Heart Disease, Titanic, Car Evaluation.
*   **Versatile Algorithms:**
    *   K-Nearest Neighbors (KNN)
    *   Decision Tree
    *   Random Forest
    *   Logistic Regression
*   **Classifier & Regressor Modes:** Most algorithms can be used for both classification and regression tasks.
*   **Hyperparameter Tuning:** Adjust key hyperparameters for each algorithm via sliders and dropdowns.
*   **Performance Metrics:**
    *   **Classification:** Training Accuracy, Testing Accuracy.
    *   **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Training R² Score, Test R² Score.
*   **Data Visualization:**
    *   **Classification:** PCA (Principal Component Analysis) plot showing data points colored by class.
    *   **Regression:**
        *   Salary Dataset: Years of Experience vs. Salary with actual data and regression line.
        *   Other Datasets: Actual vs. Predicted values plot.
*   **Dynamic UI Updates:** Hyperparameter options change based on the selected algorithm and task type.
*   **Error Handling & Progress:** Clear error messages and progress indicators during analysis.
*   **Dummy Data Generation:** Includes a script to create dummy CSV files if they are missing, allowing the app to run out-of-the-box for demonstration.

## Demo
 
 

## Prerequisites

*   Python 3.7+
*   `pip` (Python package installer)

## Setup and Installation

1.  **Clone the Repository (or download the files):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```
    If you downloaded a ZIP, extract it and navigate into the directory containing `app.py`.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    The application requires several Python libraries. Install them using:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn gradio
    ```

4.  **Dataset Preparation:**
    The application expects certain CSV files to be present in a `Dataset` folder within the same directory as `app.py`.
    *   `Dataset/Salary_dataset.csv`
    *   `Dataset/Naive-Bayes-Classification-Data.csv`
    *   `Dataset/Updated_heart_prediction.csv`
    *   `Dataset/Preprocessed Titanic Dataset.csv`
    *   `Dataset/car_evaluation.csv`

    If these files are missing, the script includes a utility to create dummy versions of these CSVs when you first run `app.py`. This allows you to test the application immediately. For actual analysis, replace these with your real datasets.

## Running the Application

1.  Navigate to the directory containing `app.py`.
2.  Ensure your virtual environment is activated (if you created one).
3.  Run the application using:
    ```bash
    python app.py
    ```
    

4.  Open your web browser and go to the local URL provided by Gradio (usually `http://127.0.0.1:7860` or a similar port).

## How to Use

1.  **Select Dataset:** Choose a dataset from the "Select Dataset" dropdown.
2.  **Select Algorithm:** Pick a machine learning algorithm from the "Select Algorithm" dropdown.
3.  **Select Algorithm Type:** Specify whether you want to use the algorithm as a "Classifier" or "Regressor".
    *   *Note: Logistic Regression is fixed as a Classifier.*
4.  **Adjust Hyperparameters:** The relevant hyperparameter tuning options for the selected algorithm and type will appear. Adjust them as needed.
5.  **Run Analysis:** Click the "🚀 Run Analysis & Visualize" button.
6.  **View Results:**
    *   **Model Insights & Performance Tab:** Displays information about the dataset, chosen algorithm, and key performance metrics.
    *   **Plot Visualization Tab:** Shows a relevant plot (PCA for classification, Actual vs. Predicted for regression).
    *   Any errors during processing will be displayed below the "Run Analysis" button.

## Code Structure

*   `app.py` (or your main script file): Contains all the Python code for the Gradio application, including:
    *   Data loading and preprocessing functions.
    *   Model definition functions.
    *   Plotting functions.
    *   The main `run_analysis` function that orchestrates the ML pipeline.
    *   Gradio UI layout and event handling logic.
    *   Dummy dataset creation utility.
*   `Dataset/` (folder): This folder should contain the CSV datasets used by the application.

## Customization and Contribution

### Adding New Datasets

1.  **Add CSV:** Place your new CSV file in the `Dataset/` folder.
2.  **Update `load_dataset` function:** Add a new `elif` condition to handle your dataset name and load it using `pd.read_csv()`.
3.  **Update `Input_output` function:** Add an `elif` condition to specify how to separate features (X) and the target variable (Y) for your new dataset. Perform any necessary preprocessing like label encoding here.
4.  **Update `dataset_names` list:** Add the display name of your new dataset to this list in the UI definition section.
5.  **Update `get_info_md` (Optional):** If your dataset has specific target class names, add a mapping for it in the `target_names_map` dictionary within `get_info_md`.
6.  **Update `choice_classifier` / `choice_regressor` (Optional):** If your dataset requires specific plotting logic, you might need to add conditions to these functions.

### Adding New Algorithms

1.  **Import:** Import the new algorithm's class from scikit-learn or other libraries.
2.  **Update Model Functions:**
    *   Add a condition to `model_classifier` (if it's a classifier) or `model_regressor` (if it's a regressor) to instantiate your new algorithm.
3.  **Update `algorithm_names` list:** Add the display name of your new algorithm.
4.  **Add Hyperparameter UI:**
    *   In the `gr.Blocks()` UI section, create a new `gr.Group()` for your algorithm's hyperparameters.
    *   Add `gr.Slider`, `gr.Dropdown`, etc., for its tunable parameters and store them in the `hyperparam_inputs` dictionary.
5.  **Update `update_algo_type_and_visibility` function:** Add conditions to show/hide the new hyperparameter group based on algorithm selection.
6.  **Update `run_analysis` function:**
    *   Add parameters for the new algorithm's hyperparameters to the function signature.
    *   Modify the `params` dictionary creation logic to include these hyperparameters when your new algorithm is selected.
    *   Ensure the new hyperparameter components are correctly added to the `ordered_hyperparam_keys` list for the `run_button.click` event.

### Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit a pull request. You can also open an issue to report bugs or suggest new features.

## Troubleshooting

*   **`NameError: name 'gr' is not defined`**: Ensure `import gradio as gr` is at the top of your Python script.
*   **`TypeError` related to `theme.set()`**: This usually means your Gradio version doesn't support certain theme customization arguments.
    *   Try upgrading Gradio: `pip install --upgrade gradio`.
    *   Simplify or remove the `.set()` call in the theme definition (e.g., just use `theme = gr.themes.Soft(...)`).
*   **File Not Found Errors for Datasets**: Make sure the `Dataset` folder exists in the same directory as your script and contains the required CSV files. The dummy data generation script should handle initial setup if files are missing.
*   **Plotting Issues/Errors**: Ensure Matplotlib and Seaborn are correctly installed. Check the console for specific error messages from these libraries.

## License

MIT License

---

Happy Hyperparameter Tuning!
```

**How to Use This README:**

1.  **Save:** Save this content as `README.md` in the root directory of your project (the same level as your `app.py` file and the `Dataset` folder).
2.  **Placeholder Screenshot:**
    *   Take a screenshot of your application running.
    *   Save it as `placeholder_screenshot.png` in the root directory, or update the image link in the README (`![HyperTuneML Screenshot](your_image_name.png)`) to the actual filename.
3.  **Repository URL:** If you are hosting this on GitHub or a similar platform, replace `<your-repository-url>` and `<repository-directory-name>` with the actual values.
4.  **File Name:** In the "Running the Application" section, ensure `python app.py` matches the actual name of your main Python script (e.g., `python app3.py`).
5.  **License (Optional):** If you want to add a specific open-source license, mention it in the "License" section and consider adding a `LICENSE` file to your repository.

This README provides a good starting point for anyone wanting to understand, use, or contribute to your project.

** Author ** 
Nishi 