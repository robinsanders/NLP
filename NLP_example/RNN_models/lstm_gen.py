#Generating jokes using an lstm model

from __future__ import print_function

import sys, os 
import numpy 
import re
import string
from nltk import word_tokenize, pos_tag

from lstm import Process_input, Lstm_char, Model_lstm_char



#Adding path to modules outside
sys.path.append(os.path.abspath(os.path.join('..', 'Data')))


from read_data import Read
from process_data import Process 

data_path = "../Data/"

reddit_name = "reddit_jokes.json"
stu_name = "stupidstuff.json"
wocka_name = "wocka.json"
se_wocka_name = "se_wocka.json"

#Jokes of string lengths outside this region will be discarded
min_length = 40
max_length = 1000

reddit_data_file = data_path + reddit_name
stu_data_file = data_path + stu_name
wocka_data_file = data_path + wocka_name
se_wocka_data_file = data_path + se_wocka_name

read_obj_wocka = Read(wocka_data_file)
read_obj_stu = Read(stu_data_file)
data_wocka = read_obj_wocka.read_file()
data_stu = read_obj_stu.read_file()

# data_wocka = data_wocka[1:1000]
fdata_wocka, vocab_wocka = Process.pre_process_char_tokenize_wocka(data_wocka, min_length, max_length)

# print(fdata_wocka[0:5])
# print(vocab_wocka)
# si = [len(li) for li in fdata_wocka]
# print(len(si))


#Model parameters and building model
input_size = len(vocab_wocka)
output_size = len(vocab_wocka)
hidden_size = 200
num_layers = 2
learning_rate = 0.01
num_epochs = 1
seq_length = min_length

model_lstm_char = Model_lstm_char(input_size, output_size, hidden_size, num_layers, learning_rate)

#Training
model_lstm_char.train(fdata_wocka, vocab_wocka, num_epochs, seq_length)



