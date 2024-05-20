import numpy as np
import matplotlib.pyplot as plt
import itertools

# Tạo confusion matrix
conf_matrix = np.array([['tp', 'fp'],
                        ['fn', 'tn']])

# Đặt nhãn cho các lớp
classes = ['Cammeo', 'Osmancik']

# Vẽ confusion matrix
plt.imshow(np.zeros_like(conf_matrix), cmap='Blues', interpolation='nearest')
plt.title('Confusion matrix')
plt.colorbar()

# Đặt các giá trị trong confusion matrix vào từng ô
for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
    plt.text(j, i, str(conf_matrix[i, j]),
             horizontalalignment="center",
             color="black")

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('Actual Rice')
plt.xlabel('Predicted Rice')
plt.tight_layout()
plt.show()
