import os
import json
from subprocess import call
import sys
import time
import itertools

class SubgraphMatching():
    def __init__(self, dataset):
        self.dataset = dataset

    def clear_and_run(self):
        dir_ = 'SubMatch/data/'
        if not os.path.exists(dir_ + self.dataset):
            os.makedirs(dir_ + self.dataset)
        call('rm -rf SubMatch/output/%s' % self.dataset, shell=True)
        call('rm -rf SubMatch/output/%s_tmp' % self.dataset, shell=True)
        call('rm -rf SubMatch/data/%s.q' % (self.dataset), shell=True)
        call('cp %s.q SubMatch/data/%s.q' % (self.dataset, self.dataset), shell=True)
        os.makedirs('SubMatch/output/%s' % self.dataset)
        os.makedirs('SubMatch/output/%s_tmp' % self.dataset)
        with open('%s.q' % self.dataset) as qfile:
            data = qfile.read()
            queries = data.split('t #\n')
            for i in range(1,len(queries)):
                with open('SubMatch/data/%s.q' % self.dataset,'w') as output:
                    output.write('t #\n'+queries[i])
                call('cat SubMatch/data/%s.q' % self.dataset, shell = True)
                print('t #\n'+queries[i])
                command = 'wine SubMatch/SubMatch.exe mode=2 data=SubMatch/data/%s.lg query=SubMatch/data/%s.q maxfreq=100000000000 stats=SubMatch/output/%s' % \
                        (self.dataset, self.dataset, self.dataset)
                call(command, shell=True)
                call('cp SubMatch/output/%s/1 SubMatch/output/%s_tmp/%s' % (self.dataset, self.dataset,i), shell=True)
                call('rm SubMatch/output/%s/1' % (self.dataset), shell=True)
                time.sleep(5)
            call('cp SubMatch/output/%s_tmp/* SubMatch/output/%s/' % (self.dataset,self.dataset), shell=True)
            call('rm -r SubMatch/output/%s_tmp' % self.dataset, shell=True)


class dataToSubgraphData:
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset == 'uspatent':
            print('uspatent!')
            self.class_id_mapping = {'c': 0, 'a': 1, 'i': 2, 'p': 3}
        elif self.dataset == 'dblp':
            self.class_id_mapping = {"a": 0, "p": 1, "v": 2}
        elif self.dataset == 'imdb':
            self.class_id_mapping = {"a": 0, "m": 1, "d": 2, "u": 3}
        elif self.dataset == 'yelp':
            self.class_id_mapping = {"b": 0, "c": 1, "u": 2}

    def data_gen(self):
        id_name_mapping = {}
        name_id_mapping = {}
        id_class_mapping = {}
        id_idx = 0
        class_idx = 0
        output_nodes = []
        # Only keep patent in final affinity matrix
        input_dir = 'data/%s/' % (self.dataset)
        # read data and change to number index
        with open(input_dir + 'node.dat') as nodes:
            for i in nodes:
                toks = i.replace('\n','').split('\t')
                id_name_mapping[id_idx] = toks[0]
                name_id_mapping[toks[0]] = id_idx
                if toks[1] not in self.class_id_mapping:
                    print(toks[1],'wrong!')
                    class_id_mapping[toks[1]] = class_idx
                    class_idx+=1
                id_class_mapping[id_idx] = self.class_id_mapping[toks[1]]
                output_nodes.append('v ' +str(id_idx) + ' ' + str(self.class_id_mapping[toks[1]]) + '\n')
                id_idx+=1
        # affinity_matrix = np.zeros((id_idx, id_idx))
        output_links = []
        missing_link = 0
        total_link = 0
        with open(input_dir + 'link.dat') as links:

            for i in links:
                total_link+=1
                toks = i.replace('\n','').split('\t')
                if toks[0] in name_id_mapping and toks[1] in name_id_mapping:
                    output_links.append('e ' + str(name_id_mapping[toks[0]])+ ' '+ str(name_id_mapping[toks[1]]) + ' 0\n')
                else:
                    missing_link+=1
        print('link not in node file:',missing_link,'; Total link:',total_link)
        with open('./SubMatch/data/%s.lg' % (self.dataset),'w') as output:
            output.write('t #\n')
            for node in output_nodes:
                output.write(node)
            for link in output_links:
                output.write(link)
if __name__ == '__main__':
    data = dataToSubgraphData(sys.argv[1])
    data.data_gen()
    match = SubgraphMatching(sys.argv[1])
    match.clear_and_run()
