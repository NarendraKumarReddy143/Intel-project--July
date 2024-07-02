import seaborn as sns
import pandas as pd
# 1)Load the Structured Data Set
# Load the Iris dataset
iris = sns.load_dataset('iris')

print(iris.head())

##2) Knowledge Representation of This Dataset
import matplotlib.pyplot as plt

# Basic statistics
print(iris.describe())

# Visualize pairplot of the dataset
sns.pairplot(iris, hue='species')
plt.show()

## 3) Draw Insights Present in the Dataset
# Box plots for feature distribution by species
plt.figure(figsize=(12, 6))
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.title('Sepal Length Distribution by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='species', y='sepal_width', data=iris)
plt.title('Sepal Width Distribution by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='species', y='petal_length', data=iris)
plt.title('Petal Length Distribution by Species')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='species', y='petal_width', data=iris)
plt.title('Petal Width Distribution by Species')
plt.show()

## 4)Correlation Analysis: Analyze the relationships between features.
# Pairplot to show relationships between features
sns.pairplot(iris, hue='species')
plt.show()

# Calculate correlation
numeric_iris = iris.drop(columns=['species'])

correlation = numeric_iris.corr()
print(correlation)

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

## 4)Display the Results/Insights Drawn from the Dataset in Pictorial Form
# Create a 2x2 grid of plots for feature distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Sepal Length Distribution by Species
sns.boxplot(x='species', y='sepal_length', data=iris, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length Distribution by Species')

# Plot 2: Sepal Width Distribution by Species
sns.boxplot(x='species', y='sepal_width', data=iris, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width Distribution by Species')

# Plot 3: Petal Length Distribution by Species
sns.boxplot(x='species', y='petal_length', data=iris, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length Distribution by Species')

# Plot 4: Petal Width Distribution by Species
sns.boxplot(x='species', y='petal_width', data=iris, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width Distribution by Species')

# Adjust layout
plt.tight_layout()
plt.show()

 


