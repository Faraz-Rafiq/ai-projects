"""
============================================================
  AL-2002 PROJECT 2026 — MODULE 4: Learning in AI
  File    : module4_iris_classifier.py
  Purpose : Implements Perceptron Learning Rule and Gradient
            Descent Delta Rule from scratch to classify Iris
            flower species. Compares both algorithms with
            different activation functions, learning rates,
            and evaluates on 80/20 train-test split.
  Dataset : UCI Iris Dataset
  Author  : 24F-0683 , 24F-0568
  Run     : streamlit run module4_iris_classifier.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
import time

# ============================================================
#  SECTION 1 — PERCEPTRON LEARNING RULE (from scratch)
# ============================================================

class PerceptronClassifier:
    """
    Implements the classic Perceptron Learning Rule from scratch.

    The Perceptron updates weights only when a misclassification
    occurs, making it an error-driven learning algorithm.
    It uses a hard threshold (step) activation function.

    Parameters
    ----------
    num_epochs   : total training iterations over the full dataset
    step_size    : learning rate — controls how much weights shift per error
    """

    def __init__(self, num_epochs=100, step_size=0.01):
        # Number of full passes through training data
        self.num_epochs = num_epochs
        # How aggressively weights are updated on each error
        self.step_size = step_size
        # Weight vector (initialized during training)
        self.weight_vector = None
        # Scalar bias term
        self.bias_term = None
        # Stores misclassification count per epoch for plotting
        self.misclassification_log = []

    def _threshold_activation(self, net_input):
        """
        Hard threshold (Heaviside) activation function.
        Returns 1 if net_input >= 0, else returns 0.
        This is the standard activation for the Perceptron model.
        """
        return 1 if net_input >= 0 else 0

    def train(self, feature_matrix, target_labels):
        """
        Trains the Perceptron using the Perceptron Learning Rule.

        Algorithm:
          1. Initialize all weights and bias to zero.
          2. For each sample, compute the predicted output.
          3. If prediction != actual label, update weights:
                w = w + step_size * (actual - predicted) * x
                b = b + step_size * (actual - predicted)
          4. Record the number of errors per epoch.

        Parameters
        ----------
        feature_matrix : 2D numpy array of shape (n_samples, n_features)
        target_labels  : 1D numpy array of binary class labels (0 or 1)
        """
        total_samples, total_features = feature_matrix.shape
        # Start with zero weights and zero bias
        self.weight_vector = np.zeros(total_features)
        self.bias_term = 0.0
        self.misclassification_log = []

        for epoch in range(self.num_epochs):
            epoch_errors = 0
            for sample_idx in range(total_samples):
                sample = feature_matrix[sample_idx]
                actual_label = target_labels[sample_idx]

                # Compute weighted sum
                net_signal = np.dot(sample, self.weight_vector) + self.bias_term
                predicted_label = self._threshold_activation(net_signal)

                # Compute error signal
                error_signal = actual_label - predicted_label

                # Only update if there was a misclassification
                if error_signal != 0:
                    self.weight_vector += self.step_size * error_signal * sample
                    self.bias_term    += self.step_size * error_signal
                    epoch_errors      += 1

            self.misclassification_log.append(epoch_errors)

    def classify(self, feature_matrix):
        """
        Generates binary predictions for a given feature matrix.
        Computes net input for each sample and applies the threshold.

        Parameters
        ----------
        feature_matrix : 2D numpy array

        Returns
        -------
        numpy array of predicted labels (0 or 1)
        """
        net_inputs = np.dot(feature_matrix, self.weight_vector) + self.bias_term
        return np.array([self._threshold_activation(z) for z in net_inputs])

    def score(self, feature_matrix, true_labels):
        """
        Computes classification accuracy as a percentage.

        Parameters
        ----------
        feature_matrix : 2D numpy array
        true_labels    : 1D array of ground-truth labels

        Returns
        -------
        float : accuracy (0 to 100)
        """
        predicted = self.classify(feature_matrix)
        return np.mean(predicted == true_labels) * 100.0


# ============================================================
#  SECTION 2 — GRADIENT DESCENT DELTA RULE (from scratch)
# ============================================================

class GradientDescentLearner:
    """
    Implements the Gradient Descent Delta Rule (Widrow-Hoff / Adaline) from scratch.

    Unlike the Perceptron, this model uses a continuous activation
    function during weight updates and minimizes Mean Squared Error
    (MSE) via batch gradient descent. Supports three activation
    functions: Sigmoid, ReLU, and Tanh.

    Parameters
    ----------
    num_epochs      : number of training iterations
    step_size       : learning rate
    activation_type : 'Sigmoid', 'ReLU', or 'Tanh'
    """

    def __init__(self, num_epochs=200, step_size=0.05, activation_type="Sigmoid"):
        self.num_epochs = num_epochs
        self.step_size = step_size
        self.activation_type = activation_type
        # Weight vector initialized at training time
        self.weight_vector = None
        # Bias term
        self.bias_term = None
        # MSE loss recorded per epoch
        self.mse_loss_log = []

    def _forward_activation(self, net_input):
        """
        Applies the chosen activation function to the net input.
        - Sigmoid : smooth S-curve, output in (0, 1)
        - ReLU    : rectified linear — passes positive values only
        - Tanh    : hyperbolic tangent, output in (-1, 1)
        """
        if self.activation_type == "Sigmoid":
            clipped = np.clip(net_input, -500, 500)   # prevent overflow
            return 1.0 / (1.0 + np.exp(-clipped))
        elif self.activation_type == "ReLU":
            return np.maximum(0.0, net_input)
        else:  # Tanh
            return np.tanh(net_input)

    def _activation_gradient(self, net_input):
        """
        Returns the derivative of the activation function.
        Used in the gradient computation during backpropagation.
        - Sigmoid derivative : sigma(x) * (1 - sigma(x))
        - ReLU derivative    : 1 where x > 0, else 0
        - Tanh derivative    : 1 - tanh(x)^2
        """
        if self.activation_type == "Sigmoid":
            activated = self._forward_activation(net_input)
            return activated * (1.0 - activated)
        elif self.activation_type == "ReLU":
            return np.where(net_input > 0, 1.0, 0.0)
        else:  # Tanh
            return 1.0 - np.tanh(net_input) ** 2

    def train(self, feature_matrix, target_labels):
        """
        Trains the model using batch Gradient Descent (Delta Rule).

        Algorithm:
          1. Initialize weights with small random values, bias = 0.
          2. For each epoch:
             a. Compute net input for all samples.
             b. Apply activation function → continuous output.
             c. Compute error = target - output.
             d. Compute delta = error * activation_derivative(net).
             e. Update weights: w += lr * X^T . delta / N
             f. Update bias:    b += lr * mean(delta)
             g. Log MSE loss.

        Parameters
        ----------
        feature_matrix : 2D numpy array (n_samples, n_features)
        target_labels  : 1D numpy array of binary labels
        """
        num_samples, num_features = feature_matrix.shape
        # Small random initialization helps break symmetry
        rng = np.random.default_rng(seed=7)
        self.weight_vector = rng.normal(0, 0.01, size=num_features)
        self.bias_term = 0.0
        self.mse_loss_log = []

        for epoch in range(self.num_epochs):
            # Forward pass
            net_input    = np.dot(feature_matrix, self.weight_vector) + self.bias_term
            output       = self._forward_activation(net_input)

            # Error between target and continuous output
            residual     = target_labels - output

            # Record MSE loss for this epoch
            mse          = np.mean(residual ** 2)
            self.mse_loss_log.append(mse)

            # Gradient of loss with respect to net input
            delta        = residual * self._activation_gradient(net_input)

            # Batch weight and bias updates
            self.weight_vector += self.step_size * np.dot(feature_matrix.T, delta) / num_samples
            self.bias_term     += self.step_size * np.mean(delta)

    def classify(self, feature_matrix):
        """
        Generates binary predictions using a 0.5 decision threshold.
        Computes the continuous activation output, then thresholds at 0.5.

        For Tanh activation the threshold is adjusted to 0.0 since its
        output range is (-1, 1).

        Parameters
        ----------
        feature_matrix : 2D numpy array

        Returns
        -------
        numpy array of binary predictions (0 or 1)
        """
        net_input = np.dot(feature_matrix, self.weight_vector) + self.bias_term
        continuous_output = self._forward_activation(net_input)
        # Tanh outputs [-1,1] so decision boundary is at 0
        cutoff = 0.0 if self.activation_type == "Tanh" else 0.5
        return np.where(continuous_output >= cutoff, 1, 0)

    def score(self, feature_matrix, true_labels):
        """
        Computes classification accuracy as a percentage.
        """
        predicted = self.classify(feature_matrix)
        return np.mean(predicted == true_labels) * 100.0


# ============================================================
#  SECTION 3 — DATA LOADING & PREPROCESSING
# ============================================================

# Hard-coded Iris dataset (avoids network dependency issues)
# Source: UCI ML Repository — iris.data
IRIS_RAW = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
4.8,3.4,1.6,0.2,Iris-setosa
4.8,3.0,1.4,0.1,Iris-setosa
4.3,3.0,1.1,0.1,Iris-setosa
5.8,4.0,1.2,0.2,Iris-setosa
5.7,4.4,1.5,0.4,Iris-setosa
5.4,3.9,1.3,0.4,Iris-setosa
5.1,3.5,1.4,0.3,Iris-setosa
5.7,3.8,1.7,0.3,Iris-setosa
5.1,3.8,1.5,0.3,Iris-setosa
5.4,3.4,1.7,0.2,Iris-setosa
5.1,3.7,1.5,0.4,Iris-setosa
4.6,3.6,1.0,0.2,Iris-setosa
5.1,3.3,1.7,0.5,Iris-setosa
4.8,3.4,1.9,0.2,Iris-setosa
5.0,3.0,1.6,0.2,Iris-setosa
5.0,3.4,1.6,0.4,Iris-setosa
5.2,3.5,1.5,0.2,Iris-setosa
5.2,3.4,1.4,0.2,Iris-setosa
4.7,3.2,1.6,0.2,Iris-setosa
4.8,3.1,1.6,0.2,Iris-setosa
5.4,3.4,1.5,0.4,Iris-setosa
5.2,4.1,1.5,0.1,Iris-setosa
5.5,4.2,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.2,Iris-setosa
5.0,3.2,1.2,0.2,Iris-setosa
5.5,3.5,1.3,0.2,Iris-setosa
4.9,3.6,1.4,0.1,Iris-setosa
4.4,3.0,1.3,0.2,Iris-setosa
5.1,3.4,1.5,0.2,Iris-setosa
5.0,3.5,1.3,0.3,Iris-setosa
4.5,2.3,1.3,0.3,Iris-setosa
4.4,3.2,1.3,0.2,Iris-setosa
5.0,3.5,1.6,0.6,Iris-setosa
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
5.7,2.8,4.5,1.3,Iris-versicolor
6.3,3.3,4.7,1.6,Iris-versicolor
4.9,2.4,3.3,1.0,Iris-versicolor
6.6,2.9,4.6,1.3,Iris-versicolor
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.5,3.9,1.1,Iris-versicolor
5.9,3.2,4.8,1.8,Iris-versicolor
6.1,2.8,4.0,1.3,Iris-versicolor
6.3,2.5,4.9,1.5,Iris-versicolor
6.1,2.8,4.7,1.2,Iris-versicolor
6.4,2.9,4.3,1.3,Iris-versicolor
6.6,3.0,4.4,1.4,Iris-versicolor
6.8,2.8,4.8,1.4,Iris-versicolor
6.7,3.0,5.0,1.7,Iris-versicolor
6.0,2.9,4.5,1.5,Iris-versicolor
5.7,2.6,3.5,1.0,Iris-versicolor
5.5,2.4,3.8,1.1,Iris-versicolor
5.5,2.4,3.7,1.0,Iris-versicolor
5.8,2.7,3.9,1.2,Iris-versicolor
6.0,2.7,5.1,1.6,Iris-versicolor
5.4,3.0,4.5,1.5,Iris-versicolor
6.0,3.4,4.5,1.6,Iris-versicolor
6.7,3.1,4.7,1.5,Iris-versicolor
6.3,2.3,4.4,1.3,Iris-versicolor
5.6,3.0,4.1,1.3,Iris-versicolor
5.5,2.5,4.0,1.3,Iris-versicolor
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
5.8,2.6,4.0,1.2,Iris-versicolor
5.0,2.3,3.3,1.0,Iris-versicolor
5.6,2.7,4.2,1.3,Iris-versicolor
5.7,3.0,4.2,1.2,Iris-versicolor
5.7,2.9,4.2,1.3,Iris-versicolor
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
6.5,3.0,5.8,2.2,Iris-virginica
7.6,3.0,6.6,2.1,Iris-virginica
4.9,2.5,4.5,1.7,Iris-virginica
7.3,2.9,6.3,1.8,Iris-virginica
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
6.5,3.2,5.1,2.0,Iris-virginica
6.4,2.7,5.3,1.9,Iris-virginica
6.8,3.0,5.5,2.1,Iris-virginica
5.7,2.5,5.0,2.0,Iris-virginica
5.8,2.8,5.1,2.4,Iris-virginica
6.4,3.2,5.3,2.3,Iris-virginica
6.5,3.0,5.5,1.8,Iris-virginica
7.7,3.8,6.7,2.2,Iris-virginica
7.7,2.6,6.9,2.3,Iris-virginica
6.0,2.2,5.0,1.5,Iris-virginica
6.9,3.2,5.7,2.3,Iris-virginica
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica
7.2,3.0,5.8,1.6,Iris-virginica
7.4,2.8,6.1,1.9,Iris-virginica
7.9,3.8,6.4,2.0,Iris-virginica
6.4,2.8,5.6,2.2,Iris-virginica
6.3,2.8,5.1,1.5,Iris-virginica
6.1,2.6,5.6,1.4,Iris-virginica
7.7,3.0,6.1,2.3,Iris-virginica
6.3,3.4,5.6,2.4,Iris-virginica
6.4,3.1,5.5,1.8,Iris-virginica
6.0,3.0,4.8,1.8,Iris-virginica
6.9,3.1,5.4,2.1,Iris-virginica
6.7,3.1,5.6,2.4,Iris-virginica
6.9,3.1,5.1,2.3,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
6.8,3.2,5.9,2.3,Iris-virginica
6.7,3.3,5.7,2.5,Iris-virginica
6.7,3.0,5.2,2.3,Iris-virginica
6.3,2.5,5.0,1.9,Iris-virginica
6.5,3.0,5.2,2.0,Iris-virginica
6.2,3.4,5.4,2.3,Iris-virginica
5.9,3.0,5.1,1.8,Iris-virginica"""


@st.cache_data
def load_iris_dataframe():
    """
    Parses the hard-coded Iris data string into a pandas DataFrame.
    Assigns column names matching the UCI standard.
    Uses Streamlit caching so parsing only happens once per session.

    Returns
    -------
    pd.DataFrame with columns:
        sepal_len, sepal_wid, petal_len, petal_wid, flower_class
    """
    import io
    col_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "flower_class"]
    df = pd.read_csv(io.StringIO(IRIS_RAW), names=col_names)
    return df


def perform_train_test_split(dataframe, test_fraction=0.2, random_seed=21):
    """
    Splits the full dataset into train and test subsets using stratified
    sampling so each species is proportionally represented in both splits.

    Parameters
    ----------
    dataframe      : full iris DataFrame
    test_fraction  : proportion reserved for testing (default 0.20 = 20%)
    random_seed    : seed for reproducibility

    Returns
    -------
    train_df, test_df : two pandas DataFrames
    """
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=dataframe["flower_class"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_binary_labels(series, positive_class):
    """
    Converts a species column into binary labels (1 / 0).
    Samples matching positive_class get label 1, all others get 0.

    Parameters
    ----------
    series         : pandas Series of species strings
    positive_class : the species string to treat as positive (1)

    Returns
    -------
    numpy array of 0s and 1s
    """
    return np.where(series == positive_class, 1, 0)


def compute_decision_grid(model_obj, x_range, y_range, grid_step=0.03):
    """
    Generates a meshgrid over the 2D feature space and predicts
    class labels for every grid point to enable decision boundary plotting.

    Parameters
    ----------
    model_obj : trained classifier with a .classify() method
    x_range   : (min, max) tuple for the first feature axis
    y_range   : (min, max) tuple for the second feature axis
    grid_step : resolution of the mesh

    Returns
    -------
    xx, yy    : meshgrid arrays
    Z         : predicted labels reshaped to match grid
    """
    x_vals = np.arange(x_range[0], x_range[1], grid_step)
    y_vals = np.arange(y_range[0], y_range[1], grid_step)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_obj.classify(grid_points).reshape(xx.shape)
    return xx, yy, Z


# ============================================================
#  SECTION 4 — STREAMLIT APPLICATION LAYOUT
# ============================================================

def render_home_page(train_df, test_df, full_df):
    """
    Renders the Home / Dataset Overview page.
    Shows dataset statistics, train/test split sizes,
    species distribution, and an interactive scatter plot.
    """
    st.title("🌸 Iris Flower Classification — Module 4")
    st.markdown("**AL-2002 Project 2026 | Learning in AI**")
    st.markdown("---")

    # Split size summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(full_df))
    c2.metric("Training Samples (80%)", len(train_df))
    c3.metric("Testing Samples (20%)", len(test_df))

    st.markdown("---")

    # Dataset tables
    tab_all, tab_train, tab_test = st.tabs(["Full Dataset (150)", "Training Set (80%)", "Test Set (20%)"])
    with tab_all:
        st.dataframe(full_df, use_container_width=True)
    with tab_train:
        st.dataframe(train_df, use_container_width=True)
    with tab_test:
        st.dataframe(test_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Scatter Plot")

    feature_cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    col_left, col_right = st.columns([1, 3])
    with col_left:
        x_feature = st.selectbox("X-Axis Feature", feature_cols, index=0)
        y_feature = st.selectbox("Y-Axis Feature", feature_cols, index=2)
    with col_right:
        fig, ax = plt.subplots(figsize=(9, 4))
        species_list = full_df["flower_class"].unique()
        color_map = {"Iris-setosa": "#e74c3c",
                     "Iris-versicolor": "#2ecc71",
                     "Iris-virginica": "#3498db"}
        for sp in species_list:
            subset = full_df[full_df["flower_class"] == sp]
            ax.scatter(subset[x_feature], subset[y_feature],
                       label=sp, color=color_map.get(sp, "gray"),
                       alpha=0.8, edgecolors="k", linewidths=0.4, s=60)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature}  vs  {y_feature}")
        ax.legend()
        st.pyplot(fig)

    # Species distribution bar chart
    st.markdown("---")
    st.subheader("Species Distribution")
    dist_fig, dist_ax = plt.subplots(figsize=(6, 3))
    counts = full_df["flower_class"].value_counts()
    dist_ax.bar(counts.index, counts.values,
                color=["#e74c3c", "#2ecc71", "#3498db"], edgecolor="k")
    dist_ax.set_ylabel("Count")
    dist_ax.set_title("Samples per Species")
    st.pyplot(dist_fig)


def render_perceptron_page(train_df, test_df):
    """
    Renders the Perceptron Learning Rule page.

    Provides controls for:
      - Target species selection (one-vs-rest binary classification)
      - Learning rate input
      - Number of epochs slider
      - Two feature selectors for 2D decision boundary visualization

    After training displays:
      - Training and test accuracy
      - Decision boundary plot
      - Misclassification count per epoch curve
    """
    st.title("🧠 Perceptron Learning Rule")
    st.markdown(
        "The Perceptron adjusts weights only when a sample is misclassified. "
        "Uses a hard **Step (threshold)** activation function."
    )
    st.markdown("---")

    feature_cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    all_species  = train_df["flower_class"].unique().tolist()

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        chosen_species = st.selectbox("Positive Class (Target Species)", all_species, key="perc_sp")
    with ctrl2:
        learning_rate  = st.number_input("Learning Rate", min_value=0.0001,
                                          max_value=1.0, value=0.01,
                                          step=0.001, format="%.4f", key="perc_lr")
    with ctrl3:
        total_epochs   = st.slider("Number of Epochs", min_value=10,
                                    max_value=500, value=100, key="perc_ep")

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        feat_x = st.selectbox("Feature for X-Axis", feature_cols, index=2, key="perc_fx")
    with feat_col2:
        feat_y = st.selectbox("Feature for Y-Axis", feature_cols, index=3, key="perc_fy")

    if feat_x == feat_y:
        st.warning("Please select two different features.")
        return

    st.markdown("---")

    if st.button("▶  Train Perceptron", use_container_width=True):
        # Prepare arrays
        X_train = train_df[[feat_x, feat_y]].values
        y_train = build_binary_labels(train_df["flower_class"], chosen_species)
        X_test  = test_df[[feat_x, feat_y]].values
        y_test  = build_binary_labels(test_df["flower_class"], chosen_species)

        # Train
        start_time = time.time()
        clf = PerceptronClassifier(num_epochs=total_epochs, step_size=learning_rate)
        clf.train(X_train, y_train)
        elapsed = time.time() - start_time

        train_accuracy = clf.score(X_train, y_train)
        test_accuracy  = clf.score(X_test,  y_test)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Accuracy",  f"{train_accuracy:.2f}%")
        m2.metric("Test Accuracy",       f"{test_accuracy:.2f}%")
        m3.metric("Training Time",       f"{elapsed*1000:.2f} ms")

        st.markdown("---")

        # Decision boundary + convergence side-by-side
        plot_left, plot_right = st.columns(2)

        with plot_left:
            st.subheader("Decision Boundary")
            x_rng = (X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5)
            y_rng = (X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
            xx, yy, Z = compute_decision_grid(clf, x_rng, y_rng)

            fig_db, ax_db = plt.subplots(figsize=(6, 5))
            ax_db.contourf(xx, yy, Z, alpha=0.25,
                           cmap=plt.cm.get_cmap("coolwarm", 2))
            # Plot positive class
            pos_mask = y_train == 1
            neg_mask = y_train == 0
            ax_db.scatter(X_train[pos_mask, 0], X_train[pos_mask, 1],
                          c="#e74c3c", label=chosen_species,
                          edgecolors="k", linewidths=0.5, s=60, zorder=3)
            ax_db.scatter(X_train[neg_mask, 0], X_train[neg_mask, 1],
                          c="#3498db", label="Others",
                          edgecolors="k", linewidths=0.5, s=60, zorder=3)
            ax_db.set_xlabel(feat_x)
            ax_db.set_ylabel(feat_y)
            ax_db.legend()
            st.pyplot(fig_db)

        with plot_right:
            st.subheader("Misclassification Convergence")
            fig_cv, ax_cv = plt.subplots(figsize=(6, 5))
            ax_cv.plot(range(1, total_epochs + 1), clf.misclassification_log,
                       color="#e74c3c", lw=2)
            ax_cv.fill_between(range(1, total_epochs + 1),
                               clf.misclassification_log,
                               alpha=0.15, color="#e74c3c")
            ax_cv.set_xlabel("Epoch")
            ax_cv.set_ylabel("Misclassifications")
            ax_cv.set_title("Errors per Epoch")
            st.pyplot(fig_cv)

        # Learned weights display
        st.markdown("---")
        st.subheader("Learned Parameters")
        pw1, pw2 = st.columns(2)
        pw1.info(f"Weight for `{feat_x}` : **{clf.weight_vector[0]:.5f}**")
        pw2.info(f"Weight for `{feat_y}` : **{clf.weight_vector[1]:.5f}**")
        st.info(f"Bias term : **{clf.bias_term:.5f}**")


def render_delta_rule_page(train_df, test_df):
    """
    Renders the Gradient Descent Delta Rule page.

    Provides controls for:
      - Target species selection
      - Activation function choice (Sigmoid / ReLU / Tanh)
      - Learning rate and epoch count
      - Feature pair for 2D visualization

    After training displays:
      - Train/test accuracy and time
      - MSE loss curve over epochs
      - Decision boundary plot
    """
    st.title("📉 Gradient Descent Delta Rule")
    st.markdown(
        "Uses continuous activation functions and minimizes **Mean Squared Error** "
        "via batch gradient descent. Supports Sigmoid, ReLU, and Tanh activations."
    )
    st.markdown("---")

    feature_cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    all_species  = train_df["flower_class"].unique().tolist()

    # ---- Controls ----
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        chosen_species  = st.selectbox("Positive Class (Target Species)", all_species, key="dr_sp")
    with ctrl2:
        activation_fn   = st.selectbox("Activation Function",
                                        ["Sigmoid", "ReLU", "Tanh"], key="dr_act")
    with ctrl3:
        learning_rate   = st.number_input("Learning Rate", min_value=0.0001,
                                           max_value=2.0, value=0.05,
                                           step=0.001, format="%.4f", key="dr_lr")

    ep_col, _, _ = st.columns(3)
    with ep_col:
        total_epochs = st.slider("Number of Epochs", 10, 1000, 200, key="dr_ep")

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        feat_x = st.selectbox("Feature for X-Axis", feature_cols, index=2, key="dr_fx")
    with feat_col2:
        feat_y = st.selectbox("Feature for Y-Axis", feature_cols, index=3, key="dr_fy")

    if feat_x == feat_y:
        st.warning("Please select two different features.")
        return

    st.markdown("---")

    if st.button("▶  Train Delta Rule Model", use_container_width=True):
        X_train = train_df[[feat_x, feat_y]].values
        y_train = build_binary_labels(train_df["flower_class"], chosen_species)
        X_test  = test_df[[feat_x, feat_y]].values
        y_test  = build_binary_labels(test_df["flower_class"], chosen_species)

        start_time = time.time()
        gd_model = GradientDescentLearner(
            num_epochs=total_epochs,
            step_size=learning_rate,
            activation_type=activation_fn
        )
        gd_model.train(X_train, y_train)
        elapsed = time.time() - start_time

        train_accuracy = gd_model.score(X_train, y_train)
        test_accuracy  = gd_model.score(X_test,  y_test)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Accuracy",  f"{train_accuracy:.2f}%")
        m2.metric("Test Accuracy",       f"{test_accuracy:.2f}%")
        m3.metric("Training Time",       f"{elapsed*1000:.2f} ms")

        st.markdown("---")

        plot_left, plot_right = st.columns(2)

        with plot_left:
            st.subheader(f"MSE Loss Curve  [{activation_fn}]")
            fig_loss, ax_loss = plt.subplots(figsize=(6, 5))
            ax_loss.plot(range(1, total_epochs + 1), gd_model.mse_loss_log,
                         color="#f39c12", lw=2)
            ax_loss.fill_between(range(1, total_epochs + 1),
                                  gd_model.mse_loss_log,
                                  alpha=0.2, color="#f39c12")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("MSE Loss")
            ax_loss.set_title("Loss Reduction over Training")
            st.pyplot(fig_loss)

        with plot_right:
            st.subheader("Decision Boundary")
            x_rng = (X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5)
            y_rng = (X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)
            xx, yy, Z = compute_decision_grid(gd_model, x_rng, y_rng)

            fig_db, ax_db = plt.subplots(figsize=(6, 5))
            ax_db.contourf(xx, yy, Z, alpha=0.25,
                           cmap=plt.cm.get_cmap("PiYG", 2))
            pos_mask = y_train == 1
            neg_mask = y_train == 0
            ax_db.scatter(X_train[pos_mask, 0], X_train[pos_mask, 1],
                          c="#27ae60", label=chosen_species,
                          edgecolors="k", linewidths=0.5, s=60, zorder=3)
            ax_db.scatter(X_train[neg_mask, 0], X_train[neg_mask, 1],
                          c="#8e44ad", label="Others",
                          edgecolors="k", linewidths=0.5, s=60, zorder=3)
            ax_db.set_xlabel(feat_x)
            ax_db.set_ylabel(feat_y)
            ax_db.legend()
            st.pyplot(fig_db)

        # Learned parameters
        st.markdown("---")
        st.subheader("Learned Parameters")
        pw1, pw2 = st.columns(2)
        pw1.info(f"Weight for `{feat_x}` : **{gd_model.weight_vector[0]:.5f}**")
        pw2.info(f"Weight for `{feat_y}` : **{gd_model.weight_vector[1]:.5f}**")
        st.info(f"Bias term : **{gd_model.bias_term:.5f}**")


def render_comparison_page(train_df, test_df):
    """
    Renders the Algorithm Comparison page.

    Runs both Perceptron and Delta Rule on identical settings
    simultaneously and places their results side-by-side, enabling
    direct comparison of accuracy, convergence speed, and decision
    boundary shape.
    """
    st.title("⚖️  Algorithm Comparison")
    st.markdown(
        "Run both algorithms on the **same data, same features, same learning rate** "
        "and compare their performance directly."
    )
    st.markdown("---")

    feature_cols = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    all_species  = train_df["flower_class"].unique().tolist()

    # Shared controls in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Shared Settings")
    target_class  = st.sidebar.selectbox("Target Species", all_species, key="cmp_sp")
    shared_lr     = st.sidebar.slider("Learning Rate", 0.0001, 1.0, 0.01,
                                       step=0.0001, format="%.4f", key="cmp_lr")
    shared_epochs = st.sidebar.slider("Epochs", 10, 1000, 300, key="cmp_ep")
    delta_act_fn  = st.sidebar.selectbox("Delta Rule Activation",
                                          ["Sigmoid", "ReLU", "Tanh"], key="cmp_act")

    fc1, fc2 = st.columns(2)
    with fc1:
        feat_x = st.selectbox("Feature X", feature_cols, index=2, key="cmp_fx")
    with fc2:
        feat_y = st.selectbox("Feature Y", feature_cols, index=3, key="cmp_fy")

    if feat_x == feat_y:
        st.warning("Please choose two different features.")
        return

    if st.button("▶  Run Both Algorithms", use_container_width=True):
        X_train = train_df[[feat_x, feat_y]].values
        y_train = build_binary_labels(train_df["flower_class"], target_class)
        X_test  = test_df[[feat_x, feat_y]].values
        y_test  = build_binary_labels(test_df["flower_class"], target_class)

        # Train Perceptron
        t0 = time.time()
        perc_model = PerceptronClassifier(num_epochs=shared_epochs, step_size=shared_lr)
        perc_model.train(X_train, y_train)
        perc_time  = (time.time() - t0) * 1000

        # Train Delta Rule
        t1 = time.time()
        gd_model = GradientDescentLearner(
            num_epochs=shared_epochs,
            step_size=shared_lr,
            activation_type=delta_act_fn
        )
        gd_model.train(X_train, y_train)
        gd_time = (time.time() - t1) * 1000

        perc_train_acc = perc_model.score(X_train, y_train)
        perc_test_acc  = perc_model.score(X_test,  y_test)
        gd_train_acc   = gd_model.score(X_train,   y_train)
        gd_test_acc    = gd_model.score(X_test,    y_test)

        # ---- Summary table ----
        st.markdown("### 📊 Performance Summary")
        summary_data = {
            "Metric":          ["Train Accuracy", "Test Accuracy", "Training Time (ms)"],
            "Perceptron":      [f"{perc_train_acc:.2f}%", f"{perc_test_acc:.2f}%", f"{perc_time:.2f}"],
            f"Delta ({delta_act_fn})": [f"{gd_train_acc:.2f}%", f"{gd_test_acc:.2f}%", f"{gd_time:.2f}"],
        }
        st.table(pd.DataFrame(summary_data))
        st.markdown("---")

        # ---- Side-by-side boundaries ----
        st.markdown("### Decision Boundaries")
        x_rng = (X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5)
        y_rng = (X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5)

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Perceptron")
            xx, yy, Z_p = compute_decision_grid(perc_model, x_rng, y_rng)
            fig_p, ax_p = plt.subplots(figsize=(5, 4))
            ax_p.contourf(xx, yy, Z_p, alpha=0.3, cmap="Blues")
            pos = y_train == 1
            ax_p.scatter(X_train[pos, 0], X_train[pos, 1],
                         c="#1a73e8", label=target_class,
                         edgecolors="k", s=50)
            ax_p.scatter(X_train[~pos, 0], X_train[~pos, 1],
                         c="#ea4335", label="Others",
                         edgecolors="k", s=50)
            ax_p.legend(fontsize=8)
            st.pyplot(fig_p)

        with right_col:
            st.subheader(f"Delta Rule  [{delta_act_fn}]")
            xx, yy, Z_d = compute_decision_grid(gd_model, x_rng, y_rng)
            fig_d, ax_d = plt.subplots(figsize=(5, 4))
            ax_d.contourf(xx, yy, Z_d, alpha=0.3, cmap="Oranges")
            ax_d.scatter(X_train[pos, 0], X_train[pos, 1],
                         c="#f9ab00", label=target_class,
                         edgecolors="k", s=50)
            ax_d.scatter(X_train[~pos, 0], X_train[~pos, 1],
                         c="#34a853", label="Others",
                         edgecolors="k", s=50)
            ax_d.legend(fontsize=8)
            st.pyplot(fig_d)

        # ---- Convergence comparison ----
        st.markdown("---")
        st.markdown("### Learning Convergence")
        cv_left, cv_right = st.columns(2)

        with cv_left:
            st.subheader("Perceptron — Misclassifications")
            fig_pe, ax_pe = plt.subplots(figsize=(5, 3))
            ax_pe.plot(range(1, shared_epochs + 1),
                       perc_model.misclassification_log,
                       color="#1a73e8", lw=2)
            ax_pe.set_xlabel("Epoch")
            ax_pe.set_ylabel("Errors")
            st.pyplot(fig_pe)

        with cv_right:
            st.subheader(f"Delta Rule [{delta_act_fn}] — MSE Loss")
            fig_dl, ax_dl = plt.subplots(figsize=(5, 3))
            ax_dl.plot(range(1, shared_epochs + 1),
                       gd_model.mse_loss_log,
                       color="#f9ab00", lw=2)
            ax_dl.set_xlabel("Epoch")
            ax_dl.set_ylabel("MSE")
            st.pyplot(fig_dl)


# ============================================================
#  SECTION 5 — ENTRY POINT
# ============================================================

def main():
    """
    Application entry point.

    Sets up page configuration, loads and splits the Iris dataset,
    stores data in Streamlit session state for persistence across
    page navigation, and routes to the correct page based on
    the sidebar selection.
    """
    st.set_page_config(
        page_title="AL-2002 Module 4 | Iris Classifier",
        page_icon="🌸",
        layout="wide"
    )

    # Load & split data once per session
    if "iris_full" not in st.session_state:
        full_df = load_iris_dataframe()
        train_df, test_df = perform_train_test_split(full_df, test_fraction=0.2)
        st.session_state["iris_full"]  = full_df
        st.session_state["iris_train"] = train_df
        st.session_state["iris_test"]  = test_df

    full_df  = st.session_state["iris_full"]
    train_df = st.session_state["iris_train"]
    test_df  = st.session_state["iris_test"]

    # Sidebar navigation
    st.sidebar.title("🌸 Module 4 — Navigation")
    st.sidebar.markdown("---")
    pages = ["🏠 Home & Dataset", "🧠 Perceptron Rule", "📉 Delta Rule (GD)", "⚖️  Compare Both"]
    selected_page = st.sidebar.radio("Go to", pages)

    st.sidebar.markdown("---")
    st.sidebar.caption("AL-2002 Project 2026")
    st.sidebar.caption("National University FAST")

    # Page routing
    if selected_page == "🏠 Home & Dataset":
        render_home_page(train_df, test_df, full_df)
    elif selected_page == "🧠 Perceptron Rule":
        render_perceptron_page(train_df, test_df)
    elif selected_page == "📉 Delta Rule (GD)":
        render_delta_rule_page(train_df, test_df)
    elif selected_page == "⚖️  Compare Both":
        render_comparison_page(train_df, test_df)


if __name__ == "__main__":
    main()
