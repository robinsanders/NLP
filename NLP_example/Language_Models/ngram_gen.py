#Generating jokes using an n-gram model

from __future__ import print_function

import sys, os 
import numpy 
import re
import string
from nltk import word_tokenize, pos_tag

from ngram import Ngram


#Adding path to modules outside
sys.path.append(os.path.abspath(os.path.join('..', 'Data')))


from read_data import Read
from process_data import Process 

data_path = "../Data/"

reddit_name = "reddit_jokes.json"
stupidstuff_name = "stupidstuff.json"
wocka_name = "wocka.json"
se_wocka_name = "se_wocka.json"

reddit_data_file = data_path + reddit_name
stupidstuff_data_file = data_path + stupidstuff_name
wocka_data_file = data_path + wocka_name
se_wocka_data_file = data_path + se_wocka_name

read_obj = Read(wocka_data_file)
data = read_obj.read_file()
pdata = Process.pre_process_wocka(data)

se_read_obj = Read(se_wocka_data_file)
se_data = se_read_obj.read_file()



fdata = Process.post_process(pdata['body'])
se_fdata = se_data['se_data']


#Uncomment to save the extracted pos from the data. It takes time to execute so dont run it all the time
# se_fdata = Process.se_save(pdata['body'], '../Data/se_wocka.json')

uwords = Process.unique_words(fdata)
se_uwords = Process.unique_words(se_fdata)


#Defining ngram smoothing types
smoothing_type_dict = {1: "add_one", 2: "add_alpha"}
n = 3
se_n = 3


init_words = "i am"

#Ngram model for generation and completion
ngram = Ngram(fdata, se_fdata, uwords, se_uwords, n, se_n, smoothing_type_dict[2], smoothing_type_dict[1])

#Ngram model for sentence segmentation
# se_ngram = Ngram(se_fdata, se_fdata, se_uwords, se_uwords, se_n, se_n, smoothing_type_dict[1])

#Generating the sentence

sent = init_words

# while(1):
for i in range(5):

    sent_list = sent.lower().split()
    next_word = ngram.gen_next_word(sent_list[-(n-1):])
    sent = sent.strip() + " " + next_word.strip()


    #Checking if it is appropriate to add a full stop here
    sent_pos = pos_tag(word_tokenize(sent))
    sent_pos = [j for (i, j) in sent_pos]
    se_next_word = ngram.gen_next_pos(sent_pos[-(se_n-1):])

    if se_next_word == '.':
        sent = sent + '.'
        break


print(sent)



# ngram.gen_next_word(prev_words.lower().split())

# str = "ab bc cdab cdab gd bab"
# substr = "ab"
# ind_list = Ngram.find_all_substring(str, substr)
# print(ind_list)
# print(Ngram.find_all_nextword(str, substr, ind_list))
