import pprint
import pandas as pd
import numpy as np
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def draw_data(df, marker, label):
	red = df.loc[df[df.columns[-1]] == 'red']
	plt.scatter(red[df.columns[0]], red[df.columns[1]], color='r',
		marker=marker,
		label=label)

	green = df.loc[df[df.columns[-1]] == 'green']
	plt.scatter(green[df.columns[0]], green[df.columns[1]], color='g',
		marker=marker,
		label=label)

def draw_svm_res(coef, inter):
	y_from_coeff = lambda x: -(x * coef[0] + inter) / coef[1]
	x = list(np.arange(-1, 12, 0.01))
	y = [y_from_coeff(el) for el in x]
	plt.plot(x, y, label='Разделяющая гиперплоскость')

def eps_regression(task_num):
	with open('data/svmdata{}.txt'.format(task_num)) as f:
		train_df = pd.read_csv(f, sep='\t') 

	train_df[train_df.columns[-1]] = pd.factorize(train_df[train_df.columns[-1]])[0]

	params = {'epsilon' : [0.1, 0.2, 0.5, 1, 10]}

	clf = GridSearchCV(svm.SVR(), params)
	clf.fit(train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]])

	print(list(params.values()))
	print(clf.cv_results_['std_train_score'])

	plt.plot(params['epsilon'], clf.cv_results_['std_train_score'])
	plt.xticks(range(len(params['epsilon'])), params['epsilon'])
	plt.show()

	return clf.cv_results_

def test_params(task_num, c=1):
	with open('data/svmdata{}.txt'.format(task_num)) as f:
		train_df = pd.read_csv(f, sep='\t') 

	with open('data/svmdata{}test.txt'.format(task_num)) as f:
		test_df = pd.read_csv(f, sep='\t')

	params = {'kernel': ['rbf', 'sigmoid', 'poly']}
	gs_train = GridSearchCV(svm.SVC(C=c), params)
	gs_train.fit(train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]])

	gs_test = GridSearchCV(svm.SVC(C=c), params)
	gs_test.fit(test_df[train_df.columns[:-1]], test_df[train_df.columns[-1]])

	return gs_train.cv_results_, gs_test.cv_results_


def perform_task(task_num,
	c=1,
	kernel='linear'):

	with open('data/svmdata{}.txt'.format(task_num)) as f:
		train_df = pd.read_csv(f, sep='\t')

	clf = svm.SVC(random_state=0,
		tol=1e-5,
		C=c,
		kernel=kernel)

	clf.fit(train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]])

	with open('data/svmdata{}test.txt'.format(task_num)) as f:
		test_df = pd.read_csv(f, sep='\t')

	acc_test = clf.score(test_df[test_df.columns[:-1]],
		test_df[test_df.columns[-1]])
	acc_train = clf.score(train_df[test_df.columns[:-1]],
		train_df[test_df.columns[-1]])

	if kernel == 'linear':
		draw_data(train_df, 'o', 'Тренировочные данные')
		draw_svm_res(clf.coef_[0], clf.intercept_)
		draw_data(test_df, '1', 'Тестовые данные')
		plt.legend()
		title = 'Линейное ядро, c={}, точность на тестовой выборке - {}, точность на обучающей выборке - {}, количество опорных векторов - {}'.format(c,
			acc_test,
			acc_train,
			clf.n_support_)

		plt.title(title)
		plt.savefig('/media/alisa/9648C53D48C51D3D/dying/4/machine/3/task_{}c{}.png'.format(task_num, c))
		plt.show()

pp = pprint.PrettyPrinter(indent=4)
#pp.pprint(test_params(2, c=1))
pp.pprint(eps_regression(2))
