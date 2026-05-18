from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def load_dataset(filepath: Path) -> pd.DataFrame:
    return pd.read_csv(filepath)


def preprocess_breast_cancer_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Get feature matrix (x) and target vector (y)
    cols_to_drop = ['id', 'diagnosis'] + [col for col in df.columns if col.startswith('Unnamed')]
    X = df.drop(columns=cols_to_drop)
    y = df['diagnosis']

    return X, y


def preprocess_diabetes_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Handle missing values using mean
    imputer = SimpleImputer(strategy='mean')
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = imputer.fit_transform(
        df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    )

    # Remove duplicates 
    df = df.drop_duplicates()

    # Get feature matrix (X) and target vector (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    return X, y


def preprocess_iris_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Get feature matrix (x) and target vector (y)
    X = df.drop(['Id', 'Species'], axis=1)
    y = df['Species']

    return X, y


def preprocess_penguins_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Handle missing values with most frequent values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Remove duplicates
    df = df.drop_duplicates()
    df = pd.get_dummies(df, columns=['island', 'sex'], dtype=int)

    # Get feature matrix (x) and target vector (y)
    X = df.drop(['species'], axis=1)
    y = df['species']

    return X, y
