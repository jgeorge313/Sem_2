#! /usr/bin/env python

from cmath import inf
import re
import string
import pathlib
import itertools
from struct import calcsize

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env


WORD_RE = re.compile(r"[\S]+")


class MRDocSim(MRJob):
    """
    A class to count word frequency in an input file.
    """

    def mapper_get_words(self, _, line):
        """

        Parameters:
            -: None
                A value parsed from input and by default it is None because the input is just raw text.
                We do not need to use this parameter.
            line: str
                each single line a file with newline stripped

            Yields:
                (key, value) pairs
        """

        # This part extracts the name of the current document being processed
        current_file = jobconf_from_env("mapreduce.map.input.file")

        # Use this doc_name as the identifier of the document
        doc_name = pathlib.Path(current_file).stem

        for word in WORD_RE.findall(line):
            # strip any punctuation
            word = word.strip(string.punctuation)

            # enforce lowercase
            word = word.lower()
            yield((word, doc_name),1)
            
    def interesting_func(self, keys, values):
         for val in values:
             yield(val,keys)

    def really_interesting_func(self,keys,value):
        doc_list = []
        word_list = []
        val_list = []
        for val1 in value:
             doc_list.append(val1[1])
             word_list.append(val1[0])
             val_list.append([val1[0],val1[1]])
        doc_set = set(doc_list)
        word_set = set(word_list)
        power_set = [(word, doc) for word in word_set for doc in doc_set]
        for val2 in val_list:
             yield(val2,keys)
        for val3 in power_set:
            yield(val3, 0)
 

    def word_count_combiner(self, keys, values):
        sum = 0
        for val in values:
            sum+=val
        yield(keys[0],(keys[1],sum))


            
   
    def calc_pair_wise_similarity_reducer(self, keys, values):
        values = list(values)
       
        if len(values)>=2:
            for i in range(len(values)):
                for j in range(len(values)):
                        yield((values[i][0],values[j][0]),min(values[i][1],values[j][1]))


   

    
    def final_similarity_reducer(self, keys, values):
        sum = 0
        for val in values:
            sum+=val
        yield(keys, sum)

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_get_words,
                reducer=self.interesting_func
               
                #educer=self.final_similarity_reduce
                #reducer=self.???
            ),

                 MRStep(
                    reducer = self.really_interesting_func#,
                    #reducer = self.word_count_combiner

              ),
                 MRStep(
                     reducer = self.word_count_combiner
        #            
                    
               ),

              MRStep(
                    reducer  = self.calc_pair_wise_similarity_reducer
                    
              ),
              
                 MRStep(
                     reducer = self.final_similarity_reducer
        #           
                    
              )
         ]


# this '__name__' == '__main__' clause is required: without it, `mrjob` will
# fail. The reason for this is because `mrjob` imports this exact same file
# several times to run the map-reduce job, and if we didn't have this
# if-clause, we'd be recursively requesting new map-reduce jobs.
if __name__ == "__main__":
    # this is how we call a Map-Reduce job in `mrjob`:
    MRDocSim.run()
