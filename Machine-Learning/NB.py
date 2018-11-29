import numpy as np

class NaiveBayes():

	def __init__(self):

		self._X_train = None
		self._y_train = None

		self._classes = None

		self._meanm = None
		self._varm = None

		self._priorlist = None


	def fit(self,X_train,y_train):

		prior = []

		self._X_train = X_train
		self._y_train = y_train

		self._classes = np.unique(_y_train)



		for i, c in enumerate(self._classes):
			
			#all data belong to which class
			#
			X_Index_one= self._X_train[np.where(self._y_train == c)]
			
			#calculate the prior probability for each class
			prior.append(X_Index_all.shape[0] / self._X_train.shape[0])
			
			#calculate the mean and variance
			X_Index_one_mean = np.mean(X_Index_one, axis = 0, keepdims = True)
			X_Index_one_var = np.var(X_Index_one, axis = 0, keepdims = True)

			#row : each class
			#store mean for each class
			mean = np.append(X_index_c_mean, axis=0)
			var = np.append(X_index_c_var, axis=0)

		self._priorlist = prior
		self._meanm = mean
		self._varm = var


	def predict(self, X_test):

		eps = 1e-10

		predict = []

		for x_sample in X_test:

			matx_sample = np.tile(x_sample,(len(self._classes),1))


			###############################Change to be the MLE#############################################

			mat_numerator = np.exp(-(matx_sample - self._meanm) ** 2 / (2 * self._varm + eps))

			mat_denominator = np.sqrt(2 * np.pi * self._varm + eps)

			#For one label,each conditional probability times
			list_log = np.sum(np.log(mat_numerator/mat_denominator),axis=1)

			#加上类先验概率的对数
			#Using the logrithm to avoid the problem of floating point underflow
			prior_class_x = list_log + np.log(self._priorlist)

			#Get the index of max probability
			prior_class_x_index = np.argmax(prior_class_x)

			#################################Change to be the MLE##################################################

			#Then find the label of this index
			classof_x = self._classes[prior_class_x_index]

			#store each
			predict.append(classof_x)

		return predict


		def get_score(self, X_test, y_test):

			j = 0

			for i in range(len(self.predict(X_test))):
				if self.predict(X_test)[i] == y_test[i]:
					j += 1


			return ('accuracy: {:.10%}'.format(j / len(y_test)))

