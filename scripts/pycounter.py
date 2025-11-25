import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/hansen/Downloads/transactionStatuses.csv')
# Read the CSV file

# Filter the data
#filtered_df = df[df['Column2'] != 'Captured']
filtered_df = df[df['Column2'] != 'Status']

# Create a count plot using seaborn
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=filtered_df, x='Column2')

# Calculate total count for percentage calculation
total = len(filtered_df)

# Add percentage labels on each bar
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.annotate(percentage, 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='bottom')

plt.title('Distribution of Transaction Types')
plt.xlabel('Transaction Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

