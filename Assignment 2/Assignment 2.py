import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv

#Part 1
print("Question 7")
Fav_color = 'Black'
print("My favorite color is", Fav_color, "length of characters are", len(Fav_color), "and the Fav_color[2] is", Fav_color[2])

print("\nQuestion 8")
Height_ft = '6 ft'
Lucky_digit = 7
print("Height is", Height_ft, "and lucky digit is", Lucky_digit)
print(type(Height_ft), type(Lucky_digit))

print("\nQuestion 9")
Hair_black = True
print(Hair_black)

print("\nQuestion 10")
X, Y, Z = 10, 11, 12
print(X, Y, Z)

print("\nQuestion 11")
Room_Num = None
if Room_Num:
	print(Room_Num)
else:
	print("None provided")

print("\nQuestion 14")
for i in range(10, 17):
	print(i)

print("\nQuestion 15")
i = 10
while i < 17:
	print(i)
	i+=1

print("\nQuestion 16")
list1 = [random.randint(6, 9) for i in range(0, 5)]
list1.append(random.randint(20, 25))
print("Second list value: %d" %list1[1], "\nList length: %d" %len(list1))
print(list1)

print("\nQuestion 17")
dict1 = {'X':21, 'Y':23, 'Z': 24}
print("dict1 is ", dict1, "\nY value is: %d" %dict1['Y'])
dict1['X'] = 100
print("X new value is: %d" %dict1['X'])
print(dict1.keys(), dict1.values())
for (k, v) in dict1.items():
	print(k, v)

print("\nQuestion 18")
Choice_List = ([['X', 'Y', 'Z'], [21, 23, 24]])
print("Choice_List is", Choice_List,"\nY value is: ",Choice_List[1][1])
Choice_List[1][0] = 100
print("X new value is: ", Choice_List[1][0])
print("Keys are: ",Choice_List[0], "and values are: ", Choice_List[1])
for i in Choice_List:
    print(i)

plt.title("Question 19")
plt.plot(Choice_List[1])
plt.xlabel('keys')
plt.ylabel('Choice_List')
plt.show()

x = (0, 1, 2)
plt.scatter(x, Choice_List[1])
plt.title("Question 20")
plt.show()

#Part 2
print("\nQuestion 21")
l=150
b=100
def Area_House(length, breadth):
	return length* breadth
print(Area_House(l, b))

print("\nQuestion 22")
my_list = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(my_list,"\n2nd row is: ", my_list[1], "\nValue of my_list[2, 3] is: ", my_list[2,3], "\nLast column is: ", my_list[:,3])

print("\nQuestion 23")
result = 0
for i in my_list[0]:
    result+=i
print(result)
column = my_list[:, 1]*my_list[:, 2]
print(column)

print("\nQuestion 24")
col = ['a', 'b', 'c', 'd']
ro = ['x', 'y', 'z']
my_matrix = pd.DataFrame(my_list, index = ro, columns = col)
print(my_matrix)

print("\nQuestion 25-30")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
My_Dataset = read_csv(url, names=names)
print("\nSample Data is: ")
print(My_Dataset.head())

print("\nSize of dataset is: ")
print(My_Dataset.shape)

print("\nData types of my_Dataset is: ")
print(My_Dataset.dtypes)

print("\nDistribution size of samples: ")
print(My_Dataset.groupby('class').size())

print("\nDescriptive statistics: ")
print(My_Dataset.describe())

print("\nSkews of My_Dataset: ")
print(My_Dataset.skew())

My_Dataset.plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.title("Question 27")
plt.show()

print("\nCorrelations of My_Dataset: ")
print(My_Dataset.corr())

My_Dataset.corr(method = 'pearson')
fig = plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(My_Dataset.corr(), vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.title("Question 29")
plt.show()

pd.plotting.scatter_matrix(My_Dataset)
plt.title("Question 30")
plt.show()