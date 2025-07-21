import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)
y_linear = 2 * x + 1
y_nonlinear1 = x**2
y_nonlinear2 = np.sin(x)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(x, y_linear, label="Linear: y=2x+1")
plt.title("Linear (Lurus)")
plt.grid()

plt.subplot(1,3,2)
plt.plot(x, y_nonlinear1, label="Non-Linear: y=x^2", color='r')
plt.title("Non-Linear (Parabola)")
plt.grid()

plt.subplot(1,3,3)
plt.plot(x, y_nonlinear2, label="Non-Linear: y=sin(x)", color='g')
plt.title("Non-Linear (Gelombang)")
plt.grid()

plt.tight_layout()
plt.show()