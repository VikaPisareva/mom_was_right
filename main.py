import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/education_vs_per_capita_personal_income.csv', encoding='utf8')

df = data.drop(['county_FIPS', 'state', 'county', 'per_capita_personal_income_2021', 'per_capita_personal_income_2020',
                'associate_degree_numbers_2016_2020', 'bachelor_degree_numbers_2016_2020',
                'associate_degree_percentage_2016_2020'], axis=1)

df1 = pd.read_csv('./data/education_vs_per_capita_personal_income.csv', delimiter=',', nrows=None)
df1 = df1.drop(['county_FIPS', 'state', 'county',
                'associate_degree_numbers_2016_2020', 'bachelor_degree_numbers_2016_2020',
                'associate_degree_percentage_2016_2020'], axis=1)


X = df.values.astype(float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

kmeans_model = None


def plotScatterMatrix(dataset, plotSize, textSize):
    dataset = dataset.select_dtypes(include=[np.number])
    dataset = dataset.dropna('columns')
    dataset = dataset[[col for col in dataset if
                       dataset[col].nunique() > 1]]
    columnNames = list(dataset)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    dataset = dataset[columnNames]
    ax = pd.plotting.scatter_matrix(dataset, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = dataset.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                          va='center', size=textSize)
    plt.suptitle('График разброса и плотности')
    plt.show()


def runElbowMethod():
    wcss = []
    for k in range(1, 11):
        kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans_model.fit(X)
        wcss.append(kmeans_model.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('K-Means Elbow Method')
    plt.xlabel('Number of clusters')
    plt.xticks([i for i in range(11)])
    plt.ylabel('WCSS')
    plt.show()


def showClusters():
    kmeans_model = KMeans(n_clusters=4, init='k-means++', random_state=0)
    kmeans_model.fit(X)
    df['cluster'] = kmeans_model.labels_
    pd.set_option('display.max_columns', None)
    print('-'*78)
    print(df.groupby('cluster').mean())
    print('-'*78)
    clusters = (0, 1, 2, 3)
    colors = ('r', 'y', 'g', 'b')
    # кластеры
    for cl, color in zip(clusters, colors):
        plt.scatter(X[df['cluster'] == cl, 0], X[df['cluster'] == cl, 1],
                    s=50, c=color, label='Cluster ' + str(cl))
    plt.scatter(kmeans_model.cluster_centers_[:, 0],
                kmeans_model.cluster_centers_[:, 1],
                s=100, c='black', marker='x', label='Centroid')
    plt.title('Clusters')
    plt.ylabel('Principal component 1')
    plt.xlabel('Principal component 2')
    plt.legend()
    plt.show()


action = -1
print('Я умею выполнять следующие действия: \n'
      'Показать график разброса и плотности\n'
      'Выполнить "метод локтя"\n'
      'Выполнить кластеризацию данных\n'
      'Анализ выполняется на основе данных о доходе на душу населения в Соединенных Штатах \n'
      'в разбивке по округам против уровня образования\n')

while not action == 0:
    action = int(input('\nПоказать график разброса и плотности (для этого введите 1)\n'
                       'Выполнить "метод локтя" (для этого введите 2)\n'
                       'Выполнить кластеризацию данных (для этого введите 3)\n'
                       'Для завершения работы программы введите 0\n'
                       'Ваше действие: '))
    match action:
        case 1:
            plotScatterMatrix(df1, 10, 9)
        case 2:
            runElbowMethod()
        case 3:
            showClusters()
print('Выполнила Виктория Переверзева')