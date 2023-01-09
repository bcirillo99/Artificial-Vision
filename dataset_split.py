import pandas as pd

df = pd.read_csv('dataframe.csv')
split = 0.2
print(len(df))
df["label"] = df["label"].astype("int32")
d = dict([(key, 0) for key in range(1,82)])
for elem in df["label"].tolist():
    d[elem]+=1
for i in range(1,82):
  x = int(d[i]*split)
  if x==0:
    d[i]=1
  else:
    d[i]=x
df_train=df
for key in d.keys():
  q = "label == "+str(key)
  x = df.query(q).sample(n=d[key])
  df_train = pd.merge(df_train,x, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
df_val = pd.merge(df,df_train, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
print("len Train: ",len(df_train))
print(df_train)
d = dict([(key, 0) for key in range(1,82)])
for elem in df_train["label"].tolist():
    d[elem]+=1
print(d)
df_train.to_csv('/content/dataframe_train.csv', index=False)
print("len Validation: ",len(df_val))
print(df_val)
d = dict([(key, 0) for key in range(1,82)])
for elem in df_val["label"].tolist():
    d[elem]+=1
print(d)
df_val.to_csv('/content/dataframe_val.csv', index=False)
