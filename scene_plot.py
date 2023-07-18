import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import numpy as np
import random
import pandas as pd
import time

df = pd.read_csv('data.csv')
Ids = set(df['TRACK_ID'].to_list())
Ids = sorted(list(Ids),key=lambda x: int(x[-4:]))
clr = list(mcolors.CSS4_COLORS.keys())
plt.figure(figsize=(10, 10))
for i, Id in enumerate(Ids):
    x = df[df['TRACK_ID'] == Id]['X'].to_numpy()
    y = df[df['TRACK_ID'] == Id]['Y'].to_numpy()
    obj = df[df['TRACK_ID'] == Id]['OBJECT_TYPE'].to_numpy()[0]
    plt.plot(x, y, label=Id[24:]+'('+obj+')', color=clr[i+6])
plt.xlabel("X")
plt.ylabel("Y", rotation=0)
plt.title("Визуализация дорожной сцены")
plt.legend()
plt.grid()
plt.savefig("scene.png")
plt.show()