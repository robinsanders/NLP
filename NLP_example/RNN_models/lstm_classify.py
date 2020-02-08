#Generating jokes using an lstm model

from __future__ import print_function

import sys, os 
import numpy 
import re
import string
from nltk import word_tokenize, pos_tag
import torch
import torch.autograd as autograd
import torch.nn as nn 

from lstm import Process_input, Model_lstm_char, Model_lstm_cl



#Adding path to modules outside
sys.path.append(os.path.abspath(os.path.join('..', 'Data')))


from read_data import Read
from process_data import Process 

data_path = "../Data/"

reddit_name = "reddit_jokes.json"
stu_name = "stupidstuff.json"
wocka_name = "wocka.json"
se_wocka_name = "se_wocka.json"


reddit_data_file = data_path + reddit_name
stu_data_file = data_path + stu_name
wocka_data_file = data_path + wocka_name
se_wocka_data_file = data_path + se_wocka_name

read_obj_wocka = Read(wocka_data_file)
read_obj_stu = Read(stu_data_file)
data_wocka = read_obj_wocka.read_file()
data_stu = read_obj_stu.read_file()


pdata_wocka = Process.pre_process_wocka(data_wocka)
fdata_wocka, fdata_wocka_cl, vocab_wocka_data, vocab_wocka_cl = Process.post_process_cl(pdata_wocka['body'], pdata_wocka['category'])

#Dividing into test and training data
fdata_train_wocka, fdata_train_wocka_cl, fdata_test_wocka, fdata_test_wocka_cl = Process.split_data(fdata_wocka, fdata_wocka_cl, 0.8)

#Model parameters and building model

input_size = 20
output_size = len(vocab_wocka_cl)
hidden_size = 10
num_layers = 1
learning_rate = 0.1
num_epochs = 5
vocab_size = len(vocab_wocka_data)

# print(vocab_wocka_cl)
# print(fdata_wocka_cl[0])
# print(len(fdata_wocka), len(fdata_wocka_cl))

# print(' '.join(fdata_wocka[158]))
# print(fdata_wocka[158])
# print(Process_input.generate_index_st(fdata_wocka[158], vocab_wocka_data))
# print(vocab_wocka_data[0:10])

# embed = nn.Embedding(len(vocab_wocka_data), input_size)
# for ind in Process_input.generate_index_st(fdata_wocka[158], vocab_wocka_data):
#     x = torch.LongTensor([ind])
#     x = autograd.Variable(x)
#     print(embed(x))
# x = autograd.Variable(torch.LongTensor(Process_input.generate_index_st(fdata_wocka[158], vocab_wocka_data)))
# print(embed(x))

model_lstm_char = Model_lstm_cl(input_size, output_size, hidden_size, num_layers, learning_rate, vocab_size)
#Training
model_lstm_char.train_cl(fdata_train_wocka, fdata_train_wocka_cl, fdata_test_wocka, fdata_test_wocka_cl, 
    vocab_wocka_data, vocab_wocka_cl, num_epochs)



