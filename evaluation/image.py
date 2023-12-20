from prettytable import PrettyTable
data = [
    ["convert_to_csr", 6.38500, 0.05],
    ["csr_matmult_maxnnz", 4396.46500, 35.74],
    ["csr_matmult_initialization", 20.74100, 0.17],
    ["csr_matmult_calculation_region", 7491.09600, 60.89],
    ["csr_matmult_putanswer_region", 364.41400, 2.96],
    ["csr_to_coo", 23.48800, 0.19],
    ["create_sparse_tensor", 0.02600, 0.00]
]

data.reverse()

import matplotlib.pyplot as plt

# Extract the test names, durations and percentages
tests = [row[0] for row in data]
durations = [row[1] for row in data]
percentages = [row[2] for row in data]

# Calculate the width of the image based on the number of test names
image_width = len(tests) * 3

# Create a figure and a set of subplots with adjusted width
fig, ax1 = plt.subplots(figsize=(image_width, 10))

# Plot the durations
ax1.barh(tests, durations, color='b')
ax1.set_ylabel('Test')
ax1.set_xlabel('Duration(ms)', color='b')
ax1.tick_params('x', colors='b')

# Create a second x-axis for the percentages
ax2 = ax1.twiny()
ax2.plot(percentages, tests, color='r', marker='o')
ax2.set_xlabel('Percentage(%)', color='r')
ax2.tick_params('x', colors='r')

# Show the plot
plt.savefig('image.png', dpi=300)
