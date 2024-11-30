import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect('spacex.db')
query = "SELECT * FROM spacex_data"
df = pd.read_sql_query(query, conn)

df['year'] = pd.to_datetime(df['date_utc']).dt.year
df['success'] = df['success'].fillna(0).astype(int)

sns.scatterplot(data=df, x='failures', y='success')
plt.title('Relationship between Failures and Success')
plt.xlabel('Failures')
plt.ylabel('Success')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

launch_success_yearly = df.groupby('year')['success'].mean()
launch_success_yearly.plot(kind='line')
plt.title('Launch Success Yearly Trend')
plt.xlabel('Year')
plt.ylabel('Success Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_dummies = pd.get_dummies(df[['rocket', 'launchpad']], drop_first=True)
df = pd.concat([df, df_dummies], axis=1)

conn.close()
