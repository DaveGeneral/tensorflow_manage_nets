import matplotlib.pyplot as plt

# Set the style globally
# Alternatives include bmh, fivethirtyeight, ggplot,
# dark_background, seaborn-deep, etc
# plt.style.use('seaborn-white')
# plt.style.use('ggplot')
# plt.style.use('bmh')
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-deep')
# plt.style.use('seaborn-paper')
plt.style.use('seaborn-whitegrid')



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times', 'Times New Roman']
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# Set an aspect ratio
width, height = plt.figaspect(4.18)
linewidth = 1
print(width, height)
fig = plt.figure(figsize=(10, 8), dpi=400)

# # Product sales plot
# ax1 = plt.subplot(221)
# ax1.bar([1, 2, 3, 4], [125, 100, 90, 110], label="Product A", width=0.8,
#             align='center')
# plt.xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4'])
# plt.xlabel('Time (FY)')
# plt.ylabel('Sales')
# # Font style isn't accessible through rcParams
# ax1.set_title("Product A sales", fontstyle='italic')

# Opportunities by age plot
# ax2 = plt.subplot(222)
# population_ages = [1,2,
#                    10,11,13,14,14,
#                    20,20,21,21,22,22,22,23,25,25,25,25,25,27,27,
#                    30,30,30,31,32,32,32,33,33,33,33,34,34,34,34,34,34,36,37,38,39,
#                    41,41,42,42,42,43,45,45,49,
#                    55,57,59,
#                    72,]
# bins = [0,10,20,30,40,50,60]
# ax2.hist(population_ages, bins, histtype='bar', rwidth=0.8)
# plt.xlabel('Age (days)')
# plt.ylabel('Closed sales')
# ax2.set_title('Opportunities age', fontstyle='italic')

# MNIST
# -----
ax3 = plt.subplot(221)
y_series = [4,6,9,13,19,28,42,63,94,141,211]
x_ae = [3,5.9068, 9.2893, 9.6357, 10.7876, 12.7876, 12.7876]
x_pca = [3.7764,3.633,4.3696,4.7597,4.1067,4.5098,18.7939,1.7925,25.5753,25.5753,25.5753]
x_dct = [3.5165,4.4513,4.7163,4.981,5.4558,5.2292,3.4751,19.0054,22.5753,25.5753,25.5753]
x_svd = [3.5072,2.9827,3.6737,4.8881,4.3186,4.8956,3.033,22.5753,25.5753,25.5753,25.5753]
x_dwt = [3.4016,4.5113,4.7749,4.4473,4.4599,4.5201,3.9403,4.6369,19.598,22.5753,25.5753]
ax3.plot([4, 28, 42, 63, 94, 141, 211], x_ae, linewidth=linewidth, linestyle=':', marker='o', label='AE')
ax3.plot(y_series, x_pca, linewidth=linewidth, linestyle='--', marker='v', label='PCA')
ax3.plot(y_series, x_dct, linewidth=linewidth, linestyle='-.', marker='s', label='DCT')
ax3.plot(y_series, x_svd, linewidth=linewidth, linestyle='-.', marker='H', label='SVD')
ax3.plot(y_series, x_dwt, linewidth=linewidth, linestyle=':', marker='P', label='DWT')
# ax3.grid(True)
plt.xlabel('Reduced Dimension')
plt.ylabel('Fractal Dimension')
plt.xticks([4, 28, 63, 94, 141, 211])
# ax3.tick_params(axis='x', pad=8)
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.4])
ax3.set_title('Dataset MNIST', fontstyle='italic')

# CIFAR10
# -------

ax4 = plt.subplot(222)
y_series = [4,6,9,13,19,28,42,63,94,141,211]
x_ae = [0.5, 0.1678, 0.1551, 10.3003, 11.401, 11.4032, 11.2067]
x_pca = [3.3653,4.0906,3.9554,1.5782,3.9994,6.2009,5.3791,6.9906,18.8888,25.5753,25.5753]
x_dct = [3.9918,5.0943,6.243,2.8814,3.2774,4.645,6.1401,15.0665,3.3972,23.5753,25.5753]
x_svd = [3.634,2.7057,4.6587,2.6081,5.9719,2.5707,6.5027,6.2395,21.8748,25.5753,25.5753]
x_dwt = [3.4213,4.8521,5.6459,2.9588,3.3272,5.485,6.7866,12.7766,14.5139,20.3658,23.5753]
ax4.plot([4, 28, 42, 63, 94, 141, 211], x_ae, linewidth=linewidth, linestyle=':', marker='o', label='AE')
ax4.plot(y_series, x_pca, linewidth=linewidth, linestyle='--', marker='v', label='PCA')
ax4.plot(y_series, x_dct, linewidth=linewidth, linestyle='-.', marker='s', label='DCT')
ax4.plot(y_series, x_svd, linewidth=linewidth, linestyle='-.', marker='H', label='SVD')
ax4.plot(y_series, x_dwt, linewidth=linewidth, linestyle=':', marker='P', label='DWT')
plt.xlabel('Reduced Dimension')
plt.ylabel('Fractal Dimension')
plt.xticks([4, 28, 63, 94, 141, 211])
# ax3.tick_params(axis='x', pad=8)
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.4])
ax4.set_title('Dataset CIFAR-10', fontstyle='italic')


# SVHN
# -----

ax5 = plt.subplot(223)
y_series = [4, 6, 9, 13, 19, 28, 42, 63, 94, 141, 160]
x_ae = [3, 11.9392, 4.0172, 9.1322, 11.4848, 14.168, 14.168]
x_pca = [3.6906, 5.1793, 6.4785, 7.2137, 12.435, 5.422, 2.4771, 28.3359, 28.3359,28.3359, 28.3359]
x_dct = [3.8896, 5.515, 6.5078, 7.5054, 15.8202, 21.736, 27.3359, 28.3359, 28.3359,28.3359, 28.3359]
x_svd = [3.8101, 5.0242, 6.5251, 7.9185, 12.2805, 5.5287, 22.7814, 28.3359, 28.3359,28.3359, 28.3359]
x_dwt = [2.9093, 4.6779, 5.5817, 7.5835, 8.0256, 14.9638, 3.4771, 25.751, 28.3359,28.3359, 28.3359]
ax5.plot([4, 28, 42, 63, 94, 141, 160], x_ae, linewidth=linewidth, linestyle=':', marker='o', label='AE')
ax5.plot(y_series, x_pca, linewidth=linewidth, linestyle='--', marker='v', label='PCA')
ax5.plot(y_series, x_dct, linewidth=linewidth, linestyle='-.', marker='s', label='DCT')
ax5.plot(y_series, x_svd, linewidth=linewidth, linestyle='-.', marker='H', label='SVD')
ax5.plot(y_series, x_dwt, linewidth=linewidth, linestyle=':', marker='P', label='DWT')
plt.xlabel('Reduced Dimension')
plt.ylabel('Fractal Dimension')
plt.xticks([4, 28, 63, 94, 141, 160])
# ax3.tick_params(axis='x', pad=8)
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.4])
ax5.set_title('Dataset SVHN', fontstyle='italic')

# AGNEWS
# ------

ax6 = plt.subplot(224)
y_series = [4,6,9,13,19,28,42,63,94,141,211,316]
x_ae = [0.3 ,0.6975, 0.7732, 9.4179, 11.7506, 0.7732, 11.6815, 11.4215]
x_pca = [2.5685,2.3968,2.8437,2.2094,3.805,4.4401,5.4467,5.8244,4.3857,24.7651,24.7651,24.7651]
x_dct = [1.9761,2.43,2.5922,2.9844,3.9418,4.3012,4.111,2.1457,3.7315,4.676,5.0242,16.3515]
x_svd = [2.448,2.7537,2.9912,3.3966,3.6956,4.8341,4.5793,4.6637,17.3139,22.4432,24.7651,24.7651]
x_dwt = [0.9801,0.9747,1.7576,1.758,2.6681,2.5881,4.0368,4.0368,2.132,2.2803,3.9624,7.6197]
ax6.plot([4, 28, 42, 63, 94, 141, 211, 316], x_ae, linewidth=linewidth, linestyle=':', marker='o', label='AE')
ax6.plot(y_series, x_pca, linewidth=linewidth, linestyle='--', marker='v', label='PCA')
ax6.plot(y_series, x_dct, linewidth=linewidth, linestyle='-.', marker='s', label='DCT')
ax6.plot(y_series, x_svd, linewidth=linewidth, linestyle='-.', marker='H', label='SVD')
ax6.plot(y_series, x_dwt, linewidth=linewidth, linestyle=':', marker='P', label='DWT')
plt.xlabel('Reduced Dimension')
plt.ylabel('Fractal Dimension')
plt.xticks([4, 28, 63, 94, 141, 211, 316])
# ax3.tick_params(axis='x', pad=8)
leg=plt.legend(loc='best', numpoints=1, fancybox=True)

# Axes alteration to put zero values inside the figure Axes
# Avoids axis white lines cutting through zero values - fivethirtyeight style
# xmin, xmax, ymin, ymax = ax3.axis()
# ax3.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.4])
ax6.set_title('Dataset AGNEWS', fontstyle='italic')


# ---------------------------------------------------
# Space plots a bit
plt.subplots_adjust(hspace=0.25, wspace=0.40)

fig.suptitle("Seaborn-white style example")
# plt.savefig('fig-ggplot.pdf', bbox_inches='tight')
plt.savefig('fig-A.pdf', bbox_inches='tight')