from __future__ import print_function

import numpy as np
import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.nn.functional as func 
import torch.optim as optim

class Process_input(object):

    def __init__(self):

        pass 

    @classmethod
    def generate_one_hot_char_single(cls, ch, vocab):

        #Generating a one hot vector for the character
        output_size = len(vocab)

        out_single = np.zeros([1, output_size])
        out_single[0, vocab.index(ch)] = 1

        return autograd.Variable(torch.FloatTensor(out_single))

    @classmethod
    def generate_one_hot_char_multi(cls, st, vocab):

        #One hot vector for each character in the list

        out_multi = []

        for ch in st:

            out_single = Process_input.generate_one_hot_char_single(ch, vocab)
            out_multi.append(out_single)

        #Converting to a variable
        out_multi = torch.cat(out_multi).view(len(st), 1, len(vocab))

        return out_multi

    @classmethod 
    def generate_index_st(cls, st, vocab):

        out = []

        for word in st:

            out.append(vocab.index(word))

        return out







#-----------------------------------------------------------------------------------------------------------------------------
#LSTM model for generation

class Lstm_char(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers):

        super(Lstm_char, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_char = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        #Mapping from hidden units to number of outputs
        self.output_map = nn.Linear(self.hidden_size, self.output_size)

        self.hidden = self.zero_hidden()

    #Zeroing out hidden state
    def zero_hidden(self):

        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)), )*self.num_layers

    #Forward pass implementation
    def forward(self, inputs):

        out_f, self.hidden = self.lstm_char(inputs, self.hidden)
        out_f = out_f[-1, :, :]
        # print(out_f.size())
        out_f = self.output_map(out_f)
        # print(out_f.size())
        out_f = func.log_softmax(out_f, dim = 1)
        # print(out_f.size())

        return out_f





class Model_lstm_char(object):

    def __init__(self, input_size, output_size, hidden_size, num_layers, learning_rate):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        #model 
        self.model = Lstm_char(self.input_size, self.output_size, self.hidden_size, self.num_layers)
        # self.model.cuda()

        #Model parameters
        self.loss_function = nn.NLLLoss()
        self.optim = optim.SGD(self.model.parameters(), lr = self.learning_rate)


    def train(self, train_data, vocab_data, num_epochs, seq_length):

        #Number of training examples and vocabulary size
        num_train = len(train_data)
        num_vocab = len(vocab_data)



        for epoch in range(num_epochs):

            for stl in train_data:

                #Each joke string

                start_ind = 0
                loss = 0

                self.model.zero_grad()
                self.model.hidden = self.model.zero_grad()


                while start_ind + seq_length < len(stl):

                    inputs = Process_input.generate_one_hot_char_multi(stl[start_ind:start_ind + seq_length], vocab_data)
                    output = vocab_data.index(stl[start_ind + seq_length])
                    output = autograd.Variable(torch.LongTensor([output]))

                    model_output = self.model(inputs)

                    # print(model_output.size(), output.size())

                    loss += self.loss_function(model_output, output)

                    start_ind += 1

                loss.backward()
                self.optim.step()

                print(loss)


                print("done")



#-----------------------------------------------------------------------------------------------------------------------------
#Lstm model for classification
class Lstm_cl(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, vocab_size):

        super(Lstm_cl, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.lstm_cl = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers)

        #Mapping from hidden units to number of outputs
        self.output_map = nn.Linear(self.hidden_size, self.output_size)

        #Defining embedding
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        self.hidden = self.zero_hidden()

    #Zeroing out hidden state
    def zero_hidden(self):

        # return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)), )*self.num_layers
        h =  autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        c =  autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        return (h, c)

    #Forward pass implementation
    def forward(self, inputs):

        #Computing embeddings
        inputs = self.embedding(inputs)
        inputs = inputs.view(-1, 1, self.input_size)

        out_f, self.hidden = self.lstm_cl(inputs, self.hidden)

        #Output is the output of the last node
        out_f = out_f[-1]
        # print(out_f.size())
        out_f = self.output_map(out_f)
        # print(out_f.size())
        out_f = func.log_softmax(out_f, dim = 1)
        # print(out_f.size())

        return out_f


class Model_lstm_cl(object):

    def __init__(self, input_size, output_size, hidden_size, num_layers, learning_rate, vocab_size):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size

        #model 
        self.model = Lstm_cl(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.vocab_size)
        # self.model.cuda()

        #Model parameters
        self.loss_function = nn.NLLLoss()
        self.optim = optim.SGD(self.model.parameters(), lr = self.learning_rate)


    def train_cl(self, train_data, train_labels, test_data, test_labels, vocab_data, vocab_cl, num_epochs):

        #Number of training examples and vocabulary size
        num_train = len(train_data)
        num_vocab = len(vocab_data)

        MODEL_SAVE_PATH = "Models/model1.pt"


        for epoch in range(num_epochs):

            for (i, stl) in enumerate(train_data):

                self.model.zero_grad()
                self.model.hidden = self.model.zero_hidden()

                #Getting indices for each word in the list. This will be used to generate embeddings
                inputs = Process_input.generate_index_st(stl, vocab_data)
                inputs = autograd.Variable(torch.LongTensor(inputs))
                output = vocab_cl.index(train_labels[i])
                output = autograd.Variable(torch.LongTensor([output]))

                try:
                    model_output = self.model(inputs)
                except:
                    print("Skipping")
                    continue
                loss = self.loss_function(model_output, output)
                loss.backward()
                self.optim.step()
                # print("Loss (Epoch " + str(epoch) + ", " + str((i*100.0)/len(train_data)) + "% ):")
                # print(loss)
                # print(inputs.size())
                
                # torch.save(self.model.state_dict(), MODEL_SAVE_PATH)

            print("Accuracy:")
            print(self.test_cl(test_data, test_labels, vocab_data, vocab_cl, MODEL_SAVE_PATH))


    def test_cl(self, test_data, labels, vocab_data, vocab_cl, MODEL_SAVE_PATH):

        num_correct = 0
        num_tot = len(test_data)


        # self.model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        for (i, stl) in enumerate(test_data):

            self.model.zero_grad()
            self.model.hidden = self.model.zero_hidden()

            inputs = Process_input.generate_index_st(stl, vocab_data)
            inputs = autograd.Variable(torch.LongTensor(inputs))
            output = vocab_cl.index(labels[i])

            try:

                model_output = self.model(inputs)

            except:
                num_tot -= 1
                continue
            # print(model_output.size())

            _, model_cl = torch.max(model_output, 1)

            # print(model_cl)

            if int(model_cl) == output:
                num_correct += 1

            # print(i)


        #Returning accuracy
        return (num_correct + 0.0)/num_tot


