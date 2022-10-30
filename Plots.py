
# importing the required modules
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
# Simple Bar Plot
x = ['original','(0.1, 0, 0)','(0.1, 0, 0.1)','(0.1, 0.1, 0)','(0.2, 0.1, 0)','(0, 0.3, 0.3)','(0.3, 0.2, 0.3)']
y = [38.833,43.92,43.625,43.1,42.3,36.7,36.7]
plt.bar(x,y,yerr=[3.47,4.159,4.897,5.18,4.584,3.64,4.6])
plt.xlabel('Weights')
plt.xticks(rotation=80)
plt.ylabel("Accuracies")
plt.title('Weight optimization results for k=1')
plt.show()



# setting the x - coordinates
#x = np.arange(1, 10, 2)
#y = [38.83333333333333,34.125,29.958333333333332,28.208333333333336,26.291666666666668]

 #   plt.title('k_NN classification using original GED')
  #  plt.xlabel(r'k')
   # plt.ylabel('Accuracy [%]')
    #plt.ylim(0,100)
   # plt.errorbar(x,y,yerr=[3.47,4.16,4.0,3.2,3.46],barsabove=True)


  #  plt.show()














# x = np.arange(1, 9, 2)
# setting the corresponding y - coordinates
# alpha = 2
# tau = 1
# sigma = 6

# y = 1/(1/(2*tau)+np.exp(-alpha*x+sigma))
# plotting the points
# plt.plot(x, y)
# plt.title(r'$c_{sigmoid}(u \rightarrow v) = \frac{1}{\frac{1}{2\tau}+exp(-\alpha||\mu_1(u)-\mu_2(v)||_p + \sigma)}$')
# plt.xlabel(r'$||\mu_1(u)-\mu_2(v)||_p$')
# plt.ylabel(r'$c_{sigmoid}(u \rightarrow v)$')
# plt.grid(True)
# plt.show()