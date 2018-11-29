import numpy as np
import random

class NN(object):

	def __init__(self, neure_num):

		#number of layers
		self.num_layers = len(neure_num)

		#number of neures for each layer
		self.neuro_num = neure_num

		#generate the biases randomly for each layer each neure
		self.biases = [np.random.randn(y,1), for y in neuro_num[1;]]

		#weight (the line in the graph)
		#each input in each layer has the same shape of last number of neure(Fig 1)
		#y --> number of line(weight) x-->number of neures so the weight for each layer (y, x)
		self.weights = [np.random.randn(y, x) for x,y in zip(neuro_num[:-1], neuro_num[1:])]


	def forward(self, input):
		#Fig 2
		for b, w in zip(self.biases, self.weights):

			ipt = sigmoid(np.dot(w, ipt) + b)

		return ipt

	def SGD(self, training_data, epochs, mini_batch_size, lr, test_data = None):

		#
		if test_data:
			n_test = len(test_data)

		n = len(training_data)

		mini_batches = []


		for j in xrange(epochs):

			#shuffle our data
			random.shuffle(training_data)

			#set mini-batch with mini_batch_size
			for k in xrange(0, n , mini_batch_size):

				mini_batches = append(training_data[k:k + mini_batch_size])

				#update the weight and bias
				for i in mini_batches:

					self.update(i, lr)

				#output the test of each epoch
				if test_data:

                	print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            	
            	else:

                	print "Epoch {0} complete".format(j)


    def update(self, mini_batch, lr):

    	# initial the partial derivate matrix
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:

        	#use bp to calculate the partial derivate
        	#use the x_train and y_train in each batch to calculate
            delta_nabla_b, delta_nabla_w = self.bp(x, y)


            # store all the derivate for each batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #for loop for each data each weight and each biases
        #the content in for is update the value for each
        #the famouse equation : w-gradient*pd
        #because of the mini batch : eta/len(mini_batch)
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]


    def bp(self, x, y):
    	
    	# initial the partial derivate matrix
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #initial the data
        activation = x
        activations = [x]

        nosig = []


        for b, w, in zip(self.biases, self.weights):

        	#next input is the last output, and the last step of last output is in ctivation function
        	z = np.dot(w, activation) + b

        	nosig.append(z)

        	activation = sigmoid(z)

        	#store the value of each neure output
        	#because of the matrix, so the sorage is all the neure output in one layer
        	#so just the output from last layer
        	activations.append(activation)

        ##################like a partial derivate####################
        # Get δ
        delta = self.cost(activations[-1], y) * sigmoid_prime(nosig[-1])
        nabla_b[-1] = delta

        # δ * last layer ouput
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        


        for l in xrange(2, self.num_layers):
            # update from last(-l) layer
            # thi layer  δ 值 update the last layer
            z = nosig[-l]

            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())


        #because we want to find how the final MSE influence the each weight
        #but the value is chain, so we said back 'spread'
        #https://blog.csdn.net/win_in_action/article/details/52704639
        #https://blog.csdn.net/guotong1988/article/details/52096724
        ############################################################

        return (nabla_b, nabla_w)



       def evaluate(self, test_data):
        # get the results
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # return the number of truely predict
        return sum(int(x == y) for (x, y) in test_results)


    	def cost(self, output_activations, y):
    		return (output_activations-y)


	def sigmoid(z):

		return 1.0/(1.0+np.exp(-z))

	# sigmoid derivate
	def sigmoid_prime(z):

		return sigmoid(z)*(1-sigmoid(z))




