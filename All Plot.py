from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(figsize=plt.figaspect(0.5))
# BaO Plot
ax = fig.add_subplot(111,projection='3d')

x, y, z1, z2, z3, z4, z5, z6, z7, r1, r2, r3, r4 =  np.loadtxt(r'C:\Users\Nick\Desktop\Project Data\Geology synthetic data unlabelled - Copy.csv', delimiter=";", unpack=True)

ax.scatter(x,y,z1, label='MnO')
ax.scatter(x,y,z2, label='FeO')
ax.scatter(x,y,z3, label='CaO')
ax.scatter(x,y,z4, label='MgO')
ax.scatter(x,y,z5, label='SiO2')
ax.scatter(x,y,z6, label='BaO')
ax.scatter(x,y,z7, label='CO2')
ax.scatter(x,y,r1, label='Ca/Mn')
ax.scatter(x,y,r1, label='Ca/Fe')
ax.scatter(x,y,r3, label='Ca/Mg')
ax.scatter(x,y,r4, label='Ca/Ba')


ax.legend()

ax.set_xlabel('$X$', fontsize=15 )#rotation=150)
ax.set_ylabel('$Y$', fontsize=15)
ax.set_zlabel(r'$Percent of Total Concentration$', fontsize=15, rotation=60)
ax.yaxis._axinfo['label']['space_factor'] = 3.0



plt.show()