import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/winequality-red.csv', sep=';')

# Check for missing values
print("Missing values:", df.isnull().sum().sum())

# Separate features (X) and target (y)
X = df.drop('quality', axis=1)  # Features (e.g., acidity, sugar, alcohol)
y = df['quality']               # Target (wine quality rating)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Scale features (important for some models, less so for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data (optional)
# pd.DataFrame(X_train_scaled).to_csv('data/X_train.csv', index=False)
# pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
# pd.DataFrame(X_test_scaled).to_csv('data/X_test.csv', index=False)
# pd.DataFrame(y_test).to_csv('data/y_test.csv', index=False)

print("Data preprocessing complete!")
