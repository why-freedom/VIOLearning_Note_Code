import matplotlib.pyplot as plt
filename = 'Lamda.txt'
X, Y = [],[]
for line in open(filename,'r'):
		value = [float(s) for s in line.split()]
		X.append(value[0])
		Y.append(value[1])

plt.plot(X,Y,'ro')
plt.title('Lamda trends')
plt.xlabel('Iter')
plt.ylabel('Lamda')
plt.show() 
