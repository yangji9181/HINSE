import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score
from sklearn import svm
import sys
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Evaluation():
    def __init__(self, eigen_vectors, dataset_name, index_i,index_j, plot_size):
        self.eigen_vectors = eigen_vectors
        # self.truth_labels = truth_labels
        self.dataset_name = dataset_name
        self.index_i = index_i
        self.index_j = index_j
        self.dim = plot_size

    def initial_read_pairwise(self, ground_truth,data_flag,selected_class):
        idx_i = int(self.index_i)
        idx_j = int(self.index_j)
        us = self.eigen_vectors
        ''' Read ground truth data '''
        patent_class_label_truth = dict()
        with open(ground_truth) as label_file:
            misscount=0
            for line in label_file:
                toks = line.replace('\n','').split('\t')
                if data_flag:
                    if 'u' in toks[0] and 'g' in toks[1]:
                        patent_class_label_truth[toks[0]] = toks[1]
                        if toks[0] not in embedded_nodes:
                            print("not in embedded nodes dict: ", toks[0])
                else:
                    patent_class_label_truth[toks[0]] = toks[1]

            print('missing groundtruth count in embedding: ', misscount)
        truth_labels = []
        oldmapping = []
        newmapping = {}
        with open('./data/%s/node.dat' %(self.dataset_name)) as nodes:
            for line in nodes:
                toks = line.replace('\n','').split('\t')
                if toks[1] == selected_class:
                    #truth_labels.append(patent_class_label_truth[toks[0]])
                    oldmapping.append(toks[0])
        with open('tmp_comparison_file.txt' ) as check:
            for line in check:
                toks = line.replace('\n','').split('\t')
                newmapping[int(toks[0])] = toks[1]
                truth_labels.append(patent_class_label_truth[toks[1]])
        if len(oldmapping) != len(newmapping): print('Wrong length match!')
    #    for key in newmapping:
    #        if newmapping[key] !=  oldmapping[key]: print('Wrong mapping',key)
        return us,np.asarray(truth_labels)

    def svm_cv(self,dataset_data, dataset_label):
        scoring = ['f1_micro','f1_macro','f1_weighted']
        for c in [10e9]:#[1,10,100,1000,10000,100000,1000000]:
            #clf = svm.SVC(kernel = 'sigmoid', C= c, random_state=0)
            clf = SGDClassifier(penalty = 'l1', tol = 1e-6)
            # clf = svm.SVC(kernel='linear',decision_function_shape='ovo', C=1, random_state=0)
            # clf = svm.LinearSVC(penalty = 'l1', dual = False)
            scores = cross_validate(clf, dataset_data, dataset_label, scoring=scoring, cv=5, return_train_score=True)
            sorted(scores.keys())
            print('C value:',c,scores)
        return scores

    def result_log(self, biglist,index, dataset_name):
        with open('./embed_plot_log/result_log_%s_%d_fwd_l1_linear' % (dataset_name,index),'w') as file:
            for size,score in biglist:
                file.write('=========================================================================\n')
                file.write(('Embedding size :'+str(size)))
                file.write("Experiment time :" + str(datetime.datetime.now() )+ '\n')
                file.write("Experiment result :" + str(score) + '\n')
                file.write("Final result : F1-Macro" + str(np.mean(score['test_f1_macro'])) + '\n')
                file.write("Final result : F1-Micro" + str(np.mean(score['test_f1_micro'])) + '\n')
                file.write("\n")

    def plot_function(self, res_list_1, res_list_2 ,path_1, path_2, dataset_name):
        shape_li = ['k', 'b', 'b', 'b', 'r', 'r', 'g', 'b','r']
        plt.plot([size for size,score in res_list_1],\
                [np.mean(score['test_f1_macro']) for size,score in res_list_1],\
                 shape_li[0])
        plt.plot([size for size,score in res_list_2],\
                [np.mean(score['test_f1_macro']) for size,score in res_list_2],\
                 shape_li[1])
        plt.xlim(0,self.dim)
        plt.ylim(0,1)
        # plt.xlabel('Embedding dimension', fontsize=25)
        # plt.ylabel('F1-Macro', fontsize=25)
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # plt.legend(fontsize=16, loc='upper right', ncol=1)
        plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
        plt.tight_layout()
        plt.savefig('./embed_plot/%s_%s_%s_pairwise_performance_.png' % (dataset_name,path_1, path_2), format='png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Plot saved to fig folder')


    def main(self):
        ground_path = './data/%s/groundtruth/name-label.txt'% self.dataset_name
        selected_class = 'a'
        with open('../metapath_%s.txt' % self.dataset_name) as paths:
            path_dict = {}
            for i,line in enumerate(paths):

                path = line.replace('\n','')
                print(i,line)
                path_dict[i] = path.upper()
            res_list_1 = []
            res_list_2 = []
            # This one time SQRT(x)
            us, truth_label = self.initial_read_pairwise(ground_path,False,selected_class)
    #        print('u',u,u.shape)
            print(path_dict[self.index_i], path_dict[self.index_j])
            for j in range(us[0].shape[0]-10,0,-10):
                cut = j
                #print(cut)
                #dataset_ = np.concatenate((nu[:,:cut],nv[:,:cut]),axis = 1)
                dataset_1 = us[0][:,cut:]
                dataset_2 = us[1][:,cut:]
                print(dataset_1.shape, truth_label.shape)
                if dataset_1.shape[1] % 100 == 0: print (dataset_1.shape[1])
                score = self.svm_cv(dataset_1, truth_label)
                print('Macro-F1:',np.mean(score['test_f1_macro']),'Micro-F1:',np.mean(score['test_f1_micro']))
                res_list_1.append((dataset_1.shape[1],dict(score)))
                score = self.svm_cv(dataset_2, truth_label)
                print('Macro-F1:',np.mean(score['test_f1_macro']),'Micro-F1:',np.mean(score['test_f1_micro']))
                res_list_2.append((dataset_2.shape[1],dict(score)))
            # print(score)
            self.result_log(res_list_1, self.index_i,self.dataset_name)
            self.result_log(res_list_2, self.index_j,self.dataset_name)
            self.plot_function(res_list_1,res_list_2,path_dict[self.index_i].upper(),path_dict[self.index_j].upper(),self.dataset_name)
