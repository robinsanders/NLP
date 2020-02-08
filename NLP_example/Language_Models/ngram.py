#Implements an N-gram language model

from __future__ import print_function

import string 
import random
from nltk import word_tokenize, pos_tag

class Ngram(object):

    #Defining smoothing type dictionary
    smoothing_type_dict = {1: "add_one", 2: "add_alpha"}

    #Constructor with 'n' and the smoothing type and the language model data
    def __init__(self, fdata, se_fdata, uwords, se_uwords, n, se_n, smoothing_type, se_smoothing_type):
        
        self.fdata = fdata
        self.se_fdata = se_fdata
        self.uwords = uwords
        self.se_uwords = se_uwords
        self.n = n 
        self.se_n = se_n
        self.smoothing_type = smoothing_type
        self.se_smoothing_type = se_smoothing_type

    @classmethod 
    def find_all_substring(cls, str, substr):

        #Computes indices of all substring in the given substring and returns the list of indices
        ind = str.find(substr, 0)
        ind_list = []

        while ind!=-1:

            ind_list.append(ind)
            ind = str.find(substr, ind+1)

        return ind_list


    @classmethod 
    def find_all_nextword(cls, str, substr, ind_list):

        #Given a string, substring and list of indices of the substring present in the string,
        #we the words following each of the substrings locations
        
        #We add a space to the string to handle the border case when the substring is near the end
        str = str + " "

        nextword_list = []
        
        for ind in ind_list:

            #We extract the word by finding the locations of the next two spaces
            end_ind = ind + len(substr)
            first_space = str.find(' ', end_ind)
            second_space = str.find(' ', first_space + 1)

            if str[first_space+1:second_space]!='':

                nextword_list.append(str[first_space+1:second_space].strip())

        return nextword_list

    def compute_ngram_prob(self, str, substr, word):

        #Computing ngram probability
        #Base numerators and denominators without smoothing
        
        base_num = str.count(substr + " " + word)
        base_den = str.count(substr)

        # if base_den!=0:
        #     print(base_den)

        #Vocabular count which is needed for smoothing
        v = str.count(word)

        if self.smoothing_type == Ngram.smoothing_type_dict[1]:

            num = base_num + 1 + 0.0
            den = base_den + v + 0.0

        if self.smoothing_type == Ngram.smoothing_type_dict[2]:

            #Defining alpha
            alpha = 0.1

            num = base_num + alpha 
            den = base_den + alpha*v      

        return num/den

    def se_compute_ngram_prob(self, str, substr, word):

        #Computing ngram probability
        #Base numerators and denominators without smoothing
        
        base_num = str.count(substr + " " + word)
        base_den = str.count(substr)

        # if base_den!=0:
        #     print(base_den)

        #Vocabular count which is needed for smoothing
        v = str.count(word)

        if self.se_smoothing_type == Ngram.smoothing_type_dict[1]:

            num = base_num + 1 + 0.0
            den = base_den + v + 0.0

        if self.se_smoothing_type == Ngram.smoothing_type_dict[2]:

            #Defining alpha
            alpha = 0.1

            num = base_num + alpha 
            den = base_den + alpha*v      

        return num/den



    def gen_next_word(self, nwords):

        #Generates the next word given the n previous words

        #Throw error if the number of words does not equal n

        if len(nwords)!=self.n-1:

            raise Exception("Number of words must be equal to 'n-1'")

        #Computing probabilities

        #To optimize the search, we combine the entire dataset into one large string and find the required substring
        combined_fdata = [word for sent_list in self.fdata for word in sent_list]
        se_combined_fdata = [word for sent_list in self.se_fdata for word in sent_list]
        combined_fdata_string = " ".join(combined_fdata)
        se_combined_fdata_string = " ".join(se_combined_fdata)
        nwords_string = " ".join(nwords)


        ind_list = Ngram.find_all_substring(combined_fdata_string, nwords_string)
        nextword_list = Ngram.find_all_nextword(combined_fdata_string, nwords_string, ind_list)

        prob_list = []

        #We now compute the probabilities of the trigrams and select the one with the highest probability
        for curr_word in nextword_list:

            prob_list.append(self.compute_ngram_prob(combined_fdata_string, nwords_string.strip(), curr_word.strip()))


        #If no matching texts are found, we choose a random word that matches the required pos
        if(len(nextword_list)==0):


            #pos tagging the previous words
            npos = pos_tag(word_tokenize(nwords_string))
            npos = [j for (i, j) in npos]
            npos_string = " ".join(npos)
            
            #Selecting the pos probable next pos


            pos_prob_list = []

            for curr_pos in self.se_uwords:

                pos_prob_list.append(self.se_compute_ngram_prob(se_combined_fdata_string, npos_string.strip(), curr_pos.strip()))

            #Creating a list of tuples, containing the pos and their probabilities
            pos_prob_tup = [(pos_prob_list[ii], self.se_uwords[ii]) for ii in range(len(self.se_uwords))]


            #sorting
            pos_prob_tup.sort()

            max_prob_pos = pos_prob_tup[0][1]

            #Picking a random word till its pos matches
            
            #If we loop for too long, then it means that the pos cannot be found. Hence we move on to the pos with
            #next higher probability
            
            num_iter = 0
            ind = 0
            
            while(1):

                num_iter += 1

                rand_word = self.uwords[random.randrange(0, len(self.uwords))]
                tmp_pos = pos_tag(word_tokenize(rand_word))
                if tmp_pos == []:
                    #No tags
                    continue
                tmp_pos = tmp_pos[0][1]

                if tmp_pos == max_prob_pos:
                    return tmp_pos

                if num_iter >=1000:
                    num_iter = 0
                    ind += 1
                    max_prob_pos = pos_prob_tup[ind][1]



            return self.uwords[random.randrange(0, len(self.uwords))]

        
        #Returning the word with maximum occurrence probability
        max_prob_ind = prob_list.index(max(prob_list))
        # print(max_prob_ind, prob_list[max_prob_ind], nextword_list[max_prob_ind])

        return nextword_list[max_prob_ind]


    def gen_next_pos(self, npos):

        #Generates the next word given the n previous words

        #Throw error if the number of words does not equal n

        if len(npos)!=self.se_n-1:

            raise Exception("Number of pos must be equal to 'n-1'")

        #Computing probabilities


        #To optimize the search, we combine the entire dataset into one large string and find the required substring
        combined_fdata = [word for sent_list in self.se_fdata for word in sent_list]
        combined_fdata_string = " ".join(combined_fdata)
        npos_string = " ".join(npos)

        # print(combined_fdata_string[1:100], " delim ", npos_string)

        prob_list = []

        #Unlike sentences, it is more efficient to search over the efficient space of all POS than all sentence occurences
        for curr_pos in self.se_uwords:

            prob_list.append(self.se_compute_ngram_prob(combined_fdata_string, npos_string.strip(), curr_pos.strip()))

        # print(prob_list)

        #As we search through all the pos and not just occurrences of pos sequences, it is guaranteed that each pos has
        #a probability and there is a maximum for one of them


        #Returning the pos with maximum occurrence probability
        max_prob_ind = prob_list.index(max(prob_list))
        # print(max_prob_ind, prob_list[max_prob_ind], nextword_list[max_prob_ind])

        return self.se_uwords[max_prob_ind]























