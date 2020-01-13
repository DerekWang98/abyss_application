import numpy as np
from sklearn.linear_model import LinearRegression

# -1 is an unspecified value to be calculated
y = np.array([25,36,45,54,72,83]).reshape(-1,1)
X = np.array(list(range(0,6))).reshape((-1,1))

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(np.array([[6]]))
print("Using a Python package:",y_pred[0][0])

# Using my own method
ones = np.array([1]*6).reshape(-1,1)
X = np.hstack((ones,X))
theta = np.array([1,1]).reshape(-1,1)

# X is m by 2, theta is 2 by 1, hyp is m by 1
def hyp(theta,X):
	return (X@theta).reshape(-1,1)


def cost_fn(theta,X,y):
	m = len(y)
	return (1/(2*m))*((hyp(theta,X)-y).sum())**2

def grad_descent(theta,alpha,iter,X,y):
	m = len(y)

	for i in range(iter):
		theta0 = theta[0].astype(float)
		theta1 = theta[1].astype(float)

		grad0 = (hyp(theta,X)-y).sum()
		grad1 = ((hyp(theta,X)-y)*X).sum()

		theta0_new = theta0-(alpha/m)*grad0
		theta1_new = theta1-(alpha/m)*grad1

		theta = np.array([theta0_new,theta1_new]).reshape(-1,1)

	return theta

final_theta = grad_descent(theta,0.1,1000,X,y)
y_pred_2 = hyp(final_theta,np.array([1,6]))

message = f"Coding from Scratch: {y_pred_2[0][0]:.1f}"
print(message)
