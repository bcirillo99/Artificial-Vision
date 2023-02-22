import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_0 = pd.read_csv("dataframe_0.csv")
df_1 = pd.read_csv("dataframe_1.csv")
df_2 = pd.read_csv("dataframe_2.csv")

w_0 = []
w_1 = []
w_2 = []
x = []
for i in range (1,82):
  x.append(i)
  s = "label == "+str(i)
  df_0_i = df_0.query(s)
  w_0.append(np.mean(np.array(df_0_i["w_col"].tolist())))
  df_1_i = df_1.query(s)
  w_1.append(np.mean(np.array(df_1_i["w_col"].tolist())))
  df_2_i = df_2.query(s)
  w_2.append(np.mean(np.array(df_2_i["w_col"].tolist())))

plt.xticks(x)
plt.xlabel("Age")
plt.ylabel("mean value of w")
fig_0 = plt.plot(x,w_0,"ro")
plt.savefig("w_0.png")
plt.close()

plt.xticks(x)
plt.xlabel("Age")
plt.ylabel("mean value of w")
fig_1 = plt.plot(x,w_1,"bo")
plt.savefig("w_1.png")
plt.close()

plt.xticks(x)
plt.xlabel("Age")
plt.ylabel("mean value of w")
fig_2 = plt.plot(x,w_2,"go")
plt.savefig("w_2.png")
plt.close()

plt.xticks(x)
plt.xlabel("Age")
plt.ylabel("mean value of w")
plt.plot(x,w_0,"r",label='w0')
plt.plot(x,w_1,"b",label='w1')
plt.plot(x,w_2,"g",label='w2')
plt.legend()
plt.savefig("w_012.png")
