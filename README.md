🚗 Used Car Price Prediction

A machine learning project to predict the resale price of used cars based on vehicle specifications, condition, mileage, and other attributes.

The project focuses on extensive data preprocessing, feature engineering, model comparison, XGBoost hyperparameter tuning, and a Mixture-of-Experts (MoE) modeling approach.

📌 Project Overview

Used-car prices depend on several interacting factors such as:

Brand and model
Vehicle age
Mileage
Engine performance
Fuel type
Transmission
Exterior and interior colors
Accident history

This project builds regression models to estimate the price of a used vehicle from these attributes.

The dataset contains 188,533 records and 13 original features, with price as the target variable.

🎯 Objective

Build a robust regression pipeline that can accurately estimate used-car prices while handling:

High-cardinality categorical variables
Missing values
Unstructured engine specifications
Non-linear relationships
Price skewness
Differences between vehicle segments
📊 Dataset
Original Features
Feature	Description
id	Unique vehicle identifier
brand	Vehicle manufacturer
model	Vehicle model
model_year	Manufacturing/model year
milage	Vehicle mileage
fuel_type	Fuel type
engine	Raw engine specification
transmission	Transmission type
ext_col	Exterior color
int_col	Interior color
accident	Accident/damage history
clean_title	Clean title indicator
price	Target variable

The initial dataset contains missing values primarily in fuel_type, accident, and clean_title.

🛠️ Data Preprocessing
1. Removing Unnecessary Features

The id and clean_title columns were removed from the modeling dataset.

2. Categorical Cleaning

Categorical variables were standardized by:

Removing leading/trailing spaces
Converting unsupported values such as – and not supported into Other
Standardizing categorical representations
3. Engine Feature Extraction

The raw engine column contains information embedded in text. Regular expressions were used to extract:

Horsepower
Engine displacement
Cylinder count
Engine configuration

For example:

420.0HP 5.0L 8 Cylinder Engine Gasoline Fuel

is transformed into structured numerical/categorical features such as:

engine_hp = 420
engine_displacement_L = 5.0
cylinder = 8
engine_config = Standard

The extracted engine features were then used instead of the original raw engine string.

4. Missing Value Treatment

Missing engine attributes were imputed using grouped medians based on related engine characteristics, followed by global median imputation where required.

5. Vehicle Age

Instead of directly using model_year, vehicle age was derived as:

model_age = 2025 - model_year

The original model_year column was then removed.

6. Transmission Standardization

Multiple transmission descriptions were consolidated into four broader categories:

Automatic
Manual
CVT
DCT

This reduced the original 52 transmission categories to four modeling categories.

7. Accident Encoding

Accident history was converted into a binary feature:

1 → No accident reported
0 → At least one accident/damage reported
🧠 Feature Engineering

Additional features were created to capture relationships that are not directly represented in the raw data.

Engine Features
hp_per_litre
Popularity Features
brand_popularity
model_popularity
brand_mean_price
Age & Mileage Features
model_age
mileage_per_year
log_model_age
Color Features
color_match
rare_ext_color
Accident Interaction
accident_age

The project uses these engineered variables along with the original categorical and numerical features.

📈 Target Transformation

Used-car prices can have a highly skewed distribution. To reduce the effect of extreme prices, the target was log-transformed:

price_per_log = np.log(price)

Predictions are converted back to the original price scale using the inverse exponential transformation during evaluation.

🔤 Categorical Encoding

Categorical features were encoded using LabelEncoder.

The following features were encoded:

brand
model
fuel_type
transmission
engine_config
ext_col
int_col

The fitted encoders were also stored for potential transformation of new data.

🧪 Train-Validation Split

The dataset was divided into:

80% Training
20% Validation

using:

train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




🤖 Models

Three baseline regression models were evaluated:

1. Linear Regression

Used as a simple baseline to establish the performance of a linear model.

2. Random Forest

A tree-based ensemble model with:

n_estimators = 500
random_state = 42
3. XGBoost

A gradient-boosted decision tree model configured with:

n_estimators = 500
learning_rate = 0.05
max_depth = 8
subsample = 0.8
colsample_bytree = 0.8

The models were evaluated using RMSE, MAE and R².

⚙️ XGBoost Hyperparameter Tuning

XGBoost was further optimized using RandomizedSearchCV.

The search explored:

max_depth
min_child_weight
subsample
colsample_bytree
learning_rate
reg_alpha
reg_lambda

The optimization objective was negative RMSE, with cross-validation used during the search.

🧩 Mixture of Experts — MoE

To capture differences between different types of vehicles, the project also implements a Mixture-of-Experts (MoE) architecture.

The approach:

Standardizes the training features.
Uses K-Means clustering to identify vehicle groups.
Trains separate XGBoost models for different clusters.
Maintains a global XGBoost model.
Combines expert and global predictions to produce the final prediction.

The MoE implementation uses K-Means clustering together with per-cluster XGBoost models and a global XGBoost model.

This allows the model to learn different pricing patterns across heterogeneous vehicle segments instead of relying on a single global model.

📏 Evaluation Metrics

The models are evaluated using:

RMSE — Root Mean Squared Error

Measures the magnitude of prediction errors while penalizing larger errors more heavily.

MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual prices.

R² Score

Measures how much of the variance in the target is explained by the model.

The final model leaderboard is sorted by RMSE.

🔄 ML Pipeline
Raw Vehicle Data
       ↓
Data Cleaning
       ↓
Missing Value Treatment
       ↓
Engine Feature Extraction
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Log Transformation of Price
       ↓
Train / Validation Split
       ↓
Baseline Models
       ↓
XGBoost Hyperparameter Tuning
       ↓
K-Means Vehicle Clustering
       ↓
Mixture-of-Experts XGBoost
       ↓
Model Evaluation
       ↓
Final Price Prediction
🗂️ Project Structure
Used-Car-Price-Prediction/
│
├── training_data.csv
├── used_car_price_prediction.ipynb
├── README.md
└── requirements.txt

Update the filenames above if your GitHub repository uses different names.

💻 Technologies Used
Python
Pandas — Data manipulation
NumPy — Numerical computation
Matplotlib — Visualization
Seaborn — Exploratory analysis
Scikit-learn — Preprocessing, clustering and ML models
XGBoost — Gradient boosting
SciPy — Hyperparameter distributions
🚀 How to Run
1. Clone the repository
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
2. Install dependencies
pip install -r requirements.txt
3. Launch the notebook
jupyter notebook

Open the project notebook and run the cells sequentially.

Make sure training_data.csv is located in the expected project directory.

📌 Key Takeaways
Raw vehicle specifications can be transformed into meaningful predictive features.
Engine information contains valuable structured signals that can be extracted from text.
Log-transforming the target helps handle price skewness.
XGBoost provides a strong non-linear modeling framework for this problem.
Hyperparameter optimization can further improve model performance.
A Mixture-of-Experts architecture can model different pricing patterns across vehicle segments.
🔮 Future Improvements

Potential improvements include:

Target encoding or native categorical handling for high-cardinality variables
More robust outlier treatment
SHAP-based model interpretability
Cross-validation-based model comparison
Automated inference pipeline for new vehicles
Deployment through a Streamlit or FastAPI application
Experimenting with CatBoost and LightGBM
Adding external market and geographic features
