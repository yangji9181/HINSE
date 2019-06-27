import numpy as np
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score,f1_score,jaccard_similarity_score,accuracy_score,make_scorer

from sklearn import svm
# from skmultilearn.adapt import MLkNN
import sys
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def initial_read( ground_truth,data_flag, index,selected_class,dataset_name, multilabel_flag):
    idx_ = int(index)
    u = np.load('./data/spectral_emed/%s_%d_u_neg.npy' % (dataset_name,idx_)).astype(np.float64)
    # ou = np.load('../practice/spectral_emed/%s_%d_u.npy' % (dataset_name,idx_))
    s = np.load('./data/spectral_emed/%s_%d_s_neg.npy' % (dataset_name,idx_))
    # os = np.load('../practice/spectral_emed/%s_%d_s.npy' % (dataset_name,idx_))
    # print(s)
    # print(os)
    # print(u)
    # print(ou)
    print('s shape:',np.diag(s).copy().shape)
    # vh = np.load('../practice/spectral_emed/%s_%d_vh_neg.npy' % (dataset_name,idx_))
    vh = u
    # Read ground truth data
    patent_class_label_truth = dict()
    labels_set = set()
    with open(ground_truth) as label_file:
        misscount=0
        for line in label_file:
            toks = line.replace('\n','').split('\t')
            if multilabel_flag:
                labels_set.add(toks[1])
                if toks[0] not in patent_class_label_truth:
                    patent_class_label_truth[toks[0]] = set()
                patent_class_label_truth[toks[0]].add(toks[1])
            else:
                patent_class_label_truth[toks[0]] = toks[1]

        print('missing groundtruth count in embedding: ', misscount)
    truth_labels = []
    newmapping = {}
    labels_set = list(labels_set)
    selected_rows = []
    if dataset_name == 'dblp':
        path_name = './doublecheck_test_%s.txt' % (dataset_name)
    else:
        path_name = './doublecheck_test_%s.txt' % (dataset_name)
    with open(path_name) as check:
        index = 0
        for line in check:
            toks = line.replace('\n','').split('\t')
            if toks[1] not in patent_class_label_truth:
                index+=1
                continue
            selected_rows.append(index)
            index+=1
            newmapping[int(toks[0])] = toks[1]
            if multilabel_flag:
                truth_labels.append(label_gen(patent_class_label_truth[toks[1]],labels_set))
            else:
                truth_labels.append(patent_class_label_truth[toks[1]])
#    for key in newmapping:
#        if newmapping[key] !=  oldmapping[key]: print('Wrong mapping',key)
    return u[selected_rows,:],s,np.dot(u,np.diag(np.sqrt(s)).copy()),np.dot(vh.T,np.diag(np.sqrt(s)).copy()) ,np.asarray(truth_labels), newmapping


def label_gen(current_labels, labels_list):
    res = []
    for label in labels_list:
        if label in current_labels:
            res.append(1)
        else:
            res.append(0)
    return res

def svm_cv(dataset_data, dataset_label, multilabel_flag):
    if multilabel_flag: return multilabel_svm_cv(dataset_data, dataset_label)
    scoring = ['f1_micro','f1_macro','f1_weighted']
    jaccard = {'jaccard_similarity_score':make_scorer(jaccard_similarity_score)}
    for c in [10e9]:#[1,10,100,1000,10000,100000,1000000]:
        # clf = svm.SVC(kernel = 'linear', random_state=0)
        # clf = SGDClassifier(penalty = 'l1', tol = 1e-7)
        # clf = svm.SVC(kernel='rbf',decision_function_shape='ovo', C=1, random_state=0)
        clf = svm.LinearSVC( dual = False)
        splits = 5 # 5-fold cross_validate
        scores = cross_validate(clf, dataset_data, dataset_label, scoring=scoring, cv=splits, return_train_score=True)
        score_tmp = cross_validate(clf, dataset_data, dataset_label, scoring=jaccard, cv=splits, return_train_score=True)
        scores['Jaccard'] = score_tmp['test_jaccard_similarity_score']
        sorted(scores.keys())

    return scores

def multilabel_svm_cv(dataset_data, dataset_label):
    scoring = ['f1_micro','f1_macro','f1_weighted']

    parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
    score = 'f1_macro'

    # clf = GridSearchCV(MLkNN(), parameters, scoring=score)
    # clf_ = MLkNN(k=1,s = 0.5)


    # clf = RandomForestClassifier(max_depth=4, random_state=0)
    # scores = cross_validate(clf, dataset_data, dataset_label, scoring=scoring, cv=5, return_train_score=True)
    # print(scores)
    # sorted(scores.keys())
    scores = {}
    kf = KFold(n_splits=5)
    macro_f1 = []
    weight_f1 = []
    jaccards = []
    acc = []
    # clf = svm.LinearSVC(penalty = 'l1', dual = False)
    clf = SGDClassifier(tol = 1e-7)
    mklnn_macro = []
    mklnn_jaccard= []
    for train_index, test_index in kf.split(dataset_data):
        X_train, X_test = dataset_data[train_index], dataset_data[test_index]
        y_train, y_test = dataset_label[train_index], dataset_label[test_index]
        label_scores = []
        comprehensive_predict = []
        # clf_.fit(X_train, y_train)
        # mklnn_predict = clf_.predict(X_test)
        # mklnn_macro.append(f1_score(y_test,mklnn_predict,average='weighted'))
        # mklnn_jaccard.append(jaccard_similarity_score(y_test,mklnn_predict))
        # print ('MLKNN',f1_score(y_test,mklnn_predict,average='weighted'), jaccard_similarity_score(y_test,mklnn_predict),accuracy_score(y_test,mklnn_predict))
        for i in range(dataset_label.shape[1]):
            clf.fit(X_train,y_train[:,i])
            prediction = clf.predict(X_test)
            comprehensive_predict.append(prediction)
            label_scores.append(f1_score(y_test[:,i],prediction))
        # macro_f1.append(np.mean(np.asarray(label_scores)))
        macro_f1.append(f1_score(y_test,np.asarray(comprehensive_predict).T,average='macro'))
        weight_f1.append(f1_score(y_test,np.asarray(comprehensive_predict).T,average='weighted'))
        jaccards.append(jaccard_similarity_score(y_test,np.asarray(comprehensive_predict).T))
        acc.append(accuracy_score(y_test,np.asarray(comprehensive_predict).T))
    scores['test_f1_macro'] = np.asarray(macro_f1)
    scores['test_f1_weighted'] = np.asarray(weight_f1)
    scores['Jaccard'] = np.asarray(jaccards)
    scores['MlKNN_f1_macro'] = np.asarray(macro_f1)
    scores['MlKNN_Jaccard'] = np.asarray(jaccards)
    # scores['MlKNN_f1_macro'] = np.asarray(mklnn_macro)
    # scores['MlKNN_Jaccard'] = np.asarray(mklnn_jaccard)
    return scores

def result_log(biglist, index, dataset_name):
    with open('../practice/embed_plot_log/result_log_%s_%d_fwd_l1_linear' % (dataset_name,index),'w') as file:
        for size,score in biglist:
            file.write('=========================================================================\n')
            file.write(('Embedding size :'+str(size)))
            file.write("Experiment time :" + str(datetime.datetime.now() )+ '\n')
            file.write("Experiment result :" + str(score) + '\n')
            file.write("Final result : F1-Macro" + str(np.mean(score['test_f1_macro'])) + '\n')
            file.write("Final result : F1-Micro" + str(np.mean(score['test_f1_micro'])) + '\n')
            file.write("\n")

def plot_function(res_list,path,index, dataset_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Embedding dimension')
    ax.set_title('Embedding size vs. Macro-F1 using Metapath: %s' % (path))
    ax.set_ylabel('F1-Macro')

    ax.plot([size for size,score in res_list],[np.mean(score['test_f1_macro']) for size,score in res_list])
    fig.savefig('practice/embed_plot/%s_%s_%d_14_l1_linear.png' % (dataset_name,path,index))



# def main():
#     ground_path = 'practice/data/%s/groundtruth/name-label.txt'%sys.argv[1]
#     selected_class = 'a'
#     with open(sys.argv[2]) as paths:
#         index = int(sys.argv[3])
#         path_dict = {}
#         for i,line in enumerate(paths):
#             print(i)
#             path = line.replace('\n','')
#             path_dict[i] = path
#         res_list = []
#         # This one time SQRT(x)
#         u,v,nu,nv,truth_label = initial_read(ground_path,False, index,selected_class,sys.argv[1])
# #        print('u',u,u.shape)
#         print('v',v,v.shape)
#         index= int(index)
#         print(path_dict[index])
#         for j in range(u.shape[0]-10,0,-10):
#             cut = j
#             #print(cut)
#             #dataset_ = np.concatenate((nu[:,:cut],nv[:,:cut]),axis = 1)
#             dataset_ = u[:,cut:]
#             print(dataset_.shape)
#             if dataset_.shape[1] % 100 == 0: print (dataset_.shape[1])
#             score = svm_cv(dataset_, truth_label)
#             print('Macro-F1:',np.mean(score['test_f1_macro']),'Micro-F1:',np.mean(score['test_f1_micro']))
#             res_list.append((dataset_.shape[1],dict(score)))
#         # print(score)
#         result_log(res_list,index,sys.argv[1])
#         plot_function(res_list,path_dict[index],index,sys.argv[1])
#         index+=1
