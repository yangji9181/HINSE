import numpy as np
import itertools
import sys
import os
import time
import tqdm
from scipy import sparse
from subprocess import call
from scipy.sparse.linalg import svds, eigs,eigsh
from scipy.linalg import fractional_matrix_power
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Mapping dictionary

class eig_analysis():
    def __init__(self, dataset_name, selected_class, class_id_mapping, input_schema,selected_idxs):
        self.processed_instancss = set()
        self.dataset_name = dataset_name
        self.selected_class = selected_class
        self.class_id_mapping = class_id_mapping
        self.selected_class_set = set()
        self.id_class_mapping = {}
        self.id_name_mapping = {}
        self.name_id_mapping = {}
        self.schema = input_schema # original nodes instead of index
        self.inverse_class_id_mapping = {}
        self.inverse_old_idx_to_new_idx = {}
        for key in class_id_mapping:
            self.inverse_class_id_mapping[class_id_mapping[key]] = key
        self.metapaths = []
        with open('../metapath_%s.txt' % dataset_name) as input_meta:
            for line in input_meta:
                meta_path = line.replace('\n','')
                self.metapaths.append(meta_path)
                print(meta_path)
        self.new_label_li = []
        self.select_idx = selected_idxs

    def graph(self):
        '''
            Read all the nodes in
        '''
        id_idx = 0
        class_idx = 0
        output_nodes = []
        # Only keep patent in final affinity matrix
        # read data and change to number index
        with open('./data/%s/node.dat' %(self.dataset_name)) as nodes:
            for i in tqdm.tqdm(nodes):
                toks = i.replace('\n','').split('\t')
                self.id_name_mapping[id_idx] = toks[0]
                self.name_id_mapping[toks[0]] = id_idx
                if toks[1] == self.selected_class:
                    self.selected_class_set.add(toks[0])
                if toks[1] not in self.class_id_mapping:
                    self.class_id_mapping[toks[1]] = class_idx
                    class_idx+=1
                self.id_class_mapping[id_idx] = self.class_id_mapping[toks[1]]
                output_nodes.append('v ' +str(id_idx) + ' ' + str(self.class_id_mapping[toks[1]]) + '\n')
                id_idx+=1
                # if toks[0] in self.schema[toks[1]]:
                #     print('!!!!! Repeated nodes')
                # else:
                #     self.schema[toks[1]][toks[0]] = set()
                if toks[0] not in self.schema[toks[1]]:
                    self.schema[toks[1]][toks[0]] = set()
        output_links = []
        with open('./data/%s/link.dat' %(self.dataset_name)) as links:
            count_r = 0
            count_c = 0
            for line in tqdm.tqdm(links):
                toks = line.replace('\n','').split('\t')
                if toks[0] not in self.name_id_mapping or toks[1] not in self.name_id_mapping:
                    count_r+=1
                    continue
                count_c+=1
                idx_0 = self.name_id_mapping[toks[0]]
                idx_1 = self.name_id_mapping[toks[1]]
                class_0 = self.inverse_class_id_mapping[self.id_class_mapping[idx_0]]
                class_1 = self.inverse_class_id_mapping[self.id_class_mapping[idx_1]]
                # print(class_0)
                # if (class_1,toks[1]) in self.schema[class_0][toks[0]]:
                #     print('!!!! Repeated Links')
                self.schema[class_0][toks[0]].add((class_1,toks[1]))
                # if (class_0,toks[0]) in schema[class_1][toks[1]]:
                #     print('!!!! Repeated Links')
                # self.schema[class_1][toks[1]].add((class_0,toks[0]))
            print('Link without nodes:', count_r,'Kept count:',count_c)

    def dfs(self,node, graph,visited,connected_components, indexing, index_num):
        if node in connected_components: return connected_components[node]
        if node in visited:
            return 0
        stack = []
        res = 1
        visited.add(node)
        stack.append(node)
        while len(stack) > 0:
            cur_node = stack.pop()
            for neigh in graph[cur_node]:
                if neigh not in visited:
                    stack.append(neigh)
                    visited.add(neigh)
        for curr in visited:
            connected_components[curr] = len(visited)
            indexing[curr] = index_num[0]
        new_inx = index_num[0]+1
        index_num.pop()
        index_num.append(new_inx)
        return len(visited)

    def final_graph_check(self,ridli,cidli,length):
        '''
            Filter out the connected component with size less than n for all graphs
        '''
        print('inside function final_graph_check')
        n = 2000
        filter_res = np.zeros((length))
        connected_component_size_li = []
        for idx in range(len(ridli)):
            print('filtering graph #',idx)
            connected_components = {}
            graph = {}
            indexing = {}
            curindex = [0]
            for a,b in zip(ridli[idx],cidli[idx]):
                if a not in graph:
                    graph[a] = set()
                if b not in graph:
                    graph[b] = set()
                graph[a].add(b)
                graph[b].add(a)
            for node in graph:
                visited = set()
                if node in connected_components: continue
                connect_component_size = self.dfs(node, graph, visited,connected_components, indexing, curindex)
                connected_component_size_li.append(connect_component_size)
                if connect_component_size > n:
                    filter_res[node]+=connect_component_size
#                print('Calculated connected componets:',len(connected_components),';Graph size:',len(graph))
        return filter_res,connected_component_size_li

    def final_graph_check_pairwise_top_five(self,ridli,cidli,length):
        '''
            Input: two chained affnity matrices.
            Output: diag_list that contains five indexes for top five
        '''
        print('Calculating top five connected components for:')
        indexings = []
        indexings.append({})
        indexings.append({})
        filter_res = np.zeros((length))
        if len(ridli) != 2: print('NOT PAIRWISE!!!!!')
        largest_connected_component = []
        largest_connected_component.append(0)
        largest_connected_component.append(0)
        for idx in range(len(ridli)):
            curindex = []
            curindex.append(0)
            connected_components = {}
            graph = {}
            indexing = {}
            for a,b in zip(ridli[idx],cidli[idx]):
                if a not in graph:
                    graph[a] = set()
                if b not in graph:
                    graph[b] = set()
                graph[a].add(b)
                graph[b].add(a)
            for node in graph:
                visited = set()
                connect_component_size = self.dfs(node, graph, visited,connected_components, indexings[idx], curindex)
                if connect_component_size > largest_connected_component[idx]:
                    largest_connected_component[idx] = connect_component_size
        common_connected_mapping = {}
        common_connected_mapping_size = {}
        print('debug:',len(indexings),len(indexings[0]),len(indexings[1]),\
         'largest connected components size in two graph:',\
          largest_connected_component[0], largest_connected_component[1] )
        for idx in indexings[0]:
            if idx not in indexings[1]: continue
            if (indexings[0][idx], indexings[1][idx]) not in common_connected_mapping:
                common_connected_mapping[(indexings[0][idx], indexings[1][idx])] = set()
                common_connected_mapping_size[(indexings[0][idx], indexings[1][idx])] = 0
            common_connected_mapping[(indexings[0][idx], indexings[1][idx])].add(idx)
            common_connected_mapping_size[(indexings[0][idx], indexings[1][idx])]+=1
        # lfp = sorted(common_connected_mapping_size.iterkeys())[0:5]
        lfp = [a for a,b in sorted(common_connected_mapping_size.iteritems(), key= lambda kv: (kv[1], kv[0]))]
        lfp = lfp[-5:]
        print('Five largest size: ',len(common_connected_mapping[lfp[0]]),\
                len(common_connected_mapping[lfp[1]]),\
                len(common_connected_mapping[lfp[2]]),\
                len(common_connected_mapping[lfp[3]]),\
                len(common_connected_mapping[lfp[4]]))
        pos_count = 0
        for i in range(len(filter_res)):
            if i not in indexings[0] or i not in indexings[1]: continue
            if (indexings[0][i],indexings[1][i]) == lfp[4]:
                filter_res[i] = 1
                pos_count+=1
            else:
                filter_res[i] = 0
        print('Should equal:',len(common_connected_mapping[lfp[4]]),'==',pos_count)
        return filter_res


    def normalized_Laplacian(self, degree_list, adjacency_mat):
        # This is a slow construction of normalized laplacian
        # Aim to acheieve normalized laplacian for all adjacency matrices
        normalized_lap = np.copy(adjacency_mat)
        print('Shape of degree_list:',degree_list.shape,'shape of adjacencymat:',adjacency_mat.shape)
        for i in range(adjacency_mat.shape[0]):
            for j in range(adjacency_mat.shape[1]):
                if (i == j) and (degree_list[0][i] != 0):
                    normalized_lap[i,j] = 1
                elif (i != j) and (adjacency_mat[i,j] != 0):
                    normalized_lap[i,j] = -adjacency_mat[i,j]/ np.sqrt(degree_list[0][i]*degree_list[0][j])
                else:
                    normalized_lap[i,j] = 0
        return normalized_lap


    def new_matrix_construct(self,diag_list,ridli,cidli,datali,idx_i,idx_j):
        '''
        Remove points in small connected components from Affinity Matrix
        '''
        # for i in range(len(ridli)):
        #     for j in range(i+1, len(ridli)):
        #         self.final_graph_check_pairwise_top_five([ridli[q] for q in [i,j]],[cidli[q] for q in [i,j]],len(diag_list[0]))
        # diag_final = self.final_graph_check(ridli,cidli,len(diag_list[0]))
        diag_final = self.final_graph_check_pairwise_top_five(ridli, cidli,len(diag_list[0]))
        self.filtering_on_HIN(diag_final,self.select_idx)
        print('Selected index:',idx_i,idx_j)
        new_idx_to_shrink_idx = {}
        new_len = np.count_nonzero(diag_final)
        id_x = 0
        for i in range(len(diag_final)):
            if diag_final[i] != 0:
                new_idx_to_shrink_idx[i] = id_x
                id_x +=1
        rid_li_2 = []
        cid_li_2 = []
        data_li_2 = []
        for idx in range(len(ridli)):
            rid_li_2.append([])
            cid_li_2.append([])
            data_li_2.append([])
            for a,b,data in zip(ridli[idx],cidli[idx],datali[idx]):
                if a in new_idx_to_shrink_idx and b in new_idx_to_shrink_idx:
                    rid_li_2[idx].append(new_idx_to_shrink_idx[a])
                    cid_li_2[idx].append(new_idx_to_shrink_idx[b])
                    data_li_2[idx].append(data)
        return rid_li_2,cid_li_2,data_li_2,new_len

    def affinity_construct(self,graph_instance,idx,whole_data_dict_li):
        if graph_instance in self.processed_instancss:
            return
        self.processed_instancss.add(graph_instance)
        graph_instance = graph_instance.split('\t')
        type_set_list = []
        type_set_dict = {}
        for instance in graph_instance:
            if self.id_class_mapping[int(instance)] not in type_set_dict:
                type_set_dict[self.id_class_mapping[int(instance)]] = set()
            type_set_dict[self.id_class_mapping[int(instance)]].add(int(instance))
        if len(whole_data_dict_li)  <= idx:
            whole_data_dict_li.append({})
        for connected_set_key in type_set_dict:
            tmp_li =  list(type_set_dict[connected_set_key])
            combinations = list(itertools.combinations(tmp_li,2))
            for x,y in combinations:
                if (x,y) not in whole_data_dict_li[len(whole_data_dict_li)-1]:
                    whole_data_dict_li[len(whole_data_dict_li)-1][(x,y)]= 0
                    whole_data_dict_li[len(whole_data_dict_li)-1][(y,x)]= 0
                whole_data_dict_li[len(whole_data_dict_li)-1][(x,y)]+=1
                whole_data_dict_li[len(whole_data_dict_li)-1][(y,x)]+=1

    def load_subgraph(self):
        '''
            Load the output of SubMatch.exe, then construct affinity matrix
        '''
        rid_li = []
        cid_li = []
        data_li = []
        whole_data_dict_li = []
        affinity_list = []
        flag = True
        file_idx = 1
        new_patent_idx = {}
        old_idx_to_new_idx = {}
        output_for_sv = {}
        start_idx = 0
        self.meta_instance_count = []
        for node in self.selected_class_set:
            new_patent_idx[node] = start_idx
            old_idx_to_new_idx[self.name_id_mapping[node]] = start_idx
            self.inverse_old_idx_to_new_idx[start_idx] = node
            output_for_sv[start_idx] = node
            start_idx+=1
        # with open('doublecheck_test.txt') as checkfile:
        #     for line in checkfile:
        #         toks = line.replace('\n','').split('\t')
        #         if output_for_sv[int(toks[0])] != toks[1]: print(output_for_sv[int(toks[0])], toks[1])
        # with open('doublecheck_test_%s.txt' % self.dataset_name,'w') as svfile:
        #     for key in output_for_sv:
        #         svfile.write(str(key)+'\t'+output_for_sv[key]+'\n')
        print('Doublecheck writing down!')
        if not os.path.exists('./homo_graph/%s/' % self.dataset_name): os.makedirs('./homo_graph/%s/' % self.dataset_name)
        # with open('./homo_graph/%s/node.dat' % self.dataset_name,'w') as nodefile:
        #     for key in output_for_sv:
        #         nodefile.write(output_for_sv[key]+'\n')
        print('original class "%s" count: %s' % (self.selected_class, start_idx))
        while flag:
            self.processed_instancss = set()
            try:
              open('./SubMatch/output/%s/'%(self.dataset_name) + str(file_idx))
            except IOError:
              print ("Finish reading %s SubMatch outputs" % (file_idx-1))
              break
            with open('./SubMatch/output/%s/'%(self.dataset_name) + str(file_idx)) as input:
                for line in tqdm.tqdm(input):
                    self.affinity_construct(line.replace('\n',''),file_idx-1,whole_data_dict_li)
            print('Metapath # %s, Total instanse: %s' %(file_idx,len(self.processed_instancss)))
            self.meta_instance_count.append(len(self.processed_instancss))
            file_idx+=1
        old_id_name_mapping = {}
        for idx in tqdm.tqdm(range(len(whole_data_dict_li))):
            new_id_name_mapping = {}
            if len(rid_li)  <= idx:
                rid_li.append([])
            if len(cid_li)  <= idx:
                cid_li.append([])
            if len(data_li)  <= idx:
                data_li.append([])
            for x,y in whole_data_dict_li[idx]:
                if self.id_name_mapping[x] in self.selected_class_set and self.id_name_mapping[y] in self.selected_class_set:
                    rid_li[idx].append(old_idx_to_new_idx[x])
                    cid_li[idx].append(old_idx_to_new_idx[y])
                    data_li[idx].append(whole_data_dict_li[idx][(x,y)])
                    new_id_name_mapping[old_idx_to_new_idx[x]] = self.id_name_mapping[x]
            if len(old_id_name_mapping) == 0:
                old_id_name_mapping = new_id_name_mapping
            else:
                less_count = 0
                for keyi in new_id_name_mapping:
                    if keyi not in old_id_name_mapping:
                        # print('Data index not mapping!',new_id_name_mapping[keyi])
                        less_count+=1
                    elif new_id_name_mapping[keyi] != old_id_name_mapping[keyi]:
                        print('Data index not mapping!',new_id_name_mapping[keyi],old_id_name_mapping[keyi])
                old_id_name_mapping = new_id_name_mapping
            # with open('./homo_graph/%s/%s_link_%s.dat' % (self.dataset_name,self.dataset_name,self.metapaths[idx]),'w') as linkfile:
            #     for x,y in whole_data_dict_li[idx]:
            #         if self.id_name_mapping[x] in self.selected_class_set and self.id_name_mapping[y] in self.selected_class_set:
            #             linkfile.write(output_for_sv[old_idx_to_new_idx[x]] + ' ' + output_for_sv[old_idx_to_new_idx[y]] + ' ' + str(whole_data_dict_li[idx][(x,y)] )+ '\n')
            if not os.path.exists('./homo_graph/%s_noweight/' % self.dataset_name): os.makedirs('./homo_graph/%s_noweight/' % self.dataset_name)
            # with open('./homo_graph/%s_noweight/%s_link_%s.dat' % (self.dataset_name,self.dataset_name,self.metapaths[idx]),'w') as linkfile:
            #     tmp_aff = sparse.csc_matrix((data_li[idx], (rid_li[idx], cid_li[idx])), shape=(start_idx, start_idx)).todense()
            #     for num_i in range(tmp_aff.shape[0]):
            #         linkfile.write(str(num_i))
            #         for num_j in range(tmp_aff.shape[1]):
            #             if tmp_aff[num_i,num_j] != 0:
            #                 linkfile.write(' ' + str(num_j))
            #         linkfile.write('\n')
        print('Finish noweight ouputing!!!!')
        # sys.exit("System interapt!")
        for i in range(len(whole_data_dict_li)):
            affinity_list.append(sparse.csc_matrix((data_li[i], (rid_li[i], cid_li[i])), shape=(start_idx, start_idx)))
        return rid_li, cid_li, data_li, affinity_list,self.meta_instance_count

    def filter_small_component(self,rid_li, cid_li, data_li,affinity_list):
        diaglist = np.ones((1,affinity_list[0].shape[0]))[0]
        self.filtering_on_HIN(diaglist,self.select_idx)
        for i in range(len(rid_li)):
            _,connect_component_size = self.final_graph_check([rid_li[i]],[cid_li[i]],len(diaglist))
            print('Size',connect_component_size)
        return affinity_list, affinity_list[0].shape[0]
        diaglist = []
        for Aff in affinity_list:
            diaglist.append(np.asarray(np.sum(Aff,axis=0))[0])
        newrid,newcid,newdata,newlen = self.new_matrix_construct(diaglist,rid_li,cid_li,data_li,idx_i,idx_j)
        affinity_list = []
        for i in range(len(cid_li)):
            affinity_list.append(sparse.csc_matrix((newdata[i], (newrid[i], newcid[i])), shape=(newlen, newlen)))
        return affinity_list, newlen

    def filter_API(self, affinity_list):
        '''
            TODO: One can modify this function to further filter the affnity matrices
        '''
        return affinity_list

    def del_helper(self, del_class, del_node):
        for end_class,end_node in self.schema[del_class][del_node]:
            self.schema[end_class][end_node].remove((del_class,del_node))
            # if len(self.schema[end_class][end_node]) == 0:
                # self.del_helper(end_class, end_node)
                # if end_node in self.schema[end_class]:
                #     self.schema[end_class].pop(end_node)
        self.schema[del_class].pop(del_node)

    def filtering_on_HIN(self, diaglist, selected_meta_idx):
        if self.dataset_name != 'dblp':
            print('Inside function filtering_on_HIN, no need\
                    for filtering right now, Skip this function')
        if self.dataset_name == 'dblp':
            selected_authors = set()
            for i in range(len(diaglist)):
                if diaglist[i] != 0:
                    selected_authors.add(self.inverse_old_idx_to_new_idx[i])
                    # print(self.id_name_mapping[i])
            print('Selected author count:', len(selected_authors))
            for author in list(self.schema['a']):
                # print(author)
                if author not in selected_authors:
                    self.del_helper('a', author)
        print('Inside filtering on HIN function')
        # collecting filtered results:
        filtered_nodes = set()
        filtered_links = set()
        filtered_author = 0
        for schema_c in self.schema:
            print(schema_c)
            for nodes in self.schema[schema_c]:
                if schema_c == "a": filtered_author+=1
                if str(nodes)+'\t'+str(schema_c)+'\n' in filtered_nodes: print('Nodes overlapping')
                filtered_nodes.add(str(nodes)+'\t'+str(schema_c)+'\n')
                for a,links in self.schema[schema_c][nodes]:
                    if str(nodes)+'\t'+str(links)+'\n' in filtered_links: print('Links overlapping')
                    filtered_links.add(str(nodes)+'\t'+str(links)+'\n')
        print('Filtered nodes:',len(filtered_nodes),'Filtered links:',len(filtered_links),'Filtered authors:',filtered_author)
        if not os.path.exists('data/%s/tmp/' % self.dataset_name): os.makedirs('data/%s/tmp/' % self.dataset_name)
        # with open('../tmp_meta.txt','w') as new_meta:
        #     for idx in selected_meta_idx:
        #         new_meta.write(self.metapaths[idx] + '\n')
        # with open('tmp_meta.txt','w') as new_meta:
        #     for idx in selected_meta_idx:
        #         new_meta.write(self.metapaths[idx] + '\n')
        # with open('data/%s/tmp/node.dat' % self.dataset_name,'w') as node_out:
        #     for line in filtered_nodes:
        #         node_out.write(line)
        # with open('data/%s/tmp/link.dat' % self.dataset_name,'w') as link_out:
        #     for line in filtered_links:
        #         link_out.write(line)
        #call('bash ../part2_local.sh tmp_meta.txt %s' % self.dataset_name,shell=True)
        #call('python3 ../node_classification_1.py %s tmp_meta.txt' % self.dataset_name,shell=True)
        new_label_li = self.read_result(selected_meta_idx)
        self.new_label_li = new_label_li
        print('New label list:',new_label_li)
    def read_result(self, selected_meta_idx):
        label_li = []
        with open('node_classification_log.txt') as target:
            content = target.read()
            results = content.split('=========================================================================')
            for i in range(len(selected_meta_idx)):
                print('selfidx',selected_meta_idx[i])
                result_idx = len(results)-selected_meta_idx[i] - 1
                meta_idx = selected_meta_idx[len(selected_meta_idx)-i-1]
                result = results[result_idx]
                meta_path = self.metapaths[meta_idx]
                f1_micro_1 = result.split('F1-Micro')
                f1_micro_2 = f1_micro_1[1].split('\n')
                print(f1_micro_2)
                f1_micro_score = round(float(f1_micro_2[0]),3)
                f1_macro_1 = result.split('F1-Macro')
                f1_macro_2 = f1_macro_1[1].split('\n')
                f1_macro_score = round(float(f1_macro_2[0]),3)
                label = '%s F1-Micro: %s F1-Macro: %s' % (meta_path, f1_micro_score,f1_macro_score)
                label_li = [label] + label_li
        return label_li

    def output_filtered_HIN(self):
        print('Outputing filtered HIN, feed for ESim embedding')
    def check_symmtrix(self, A):
        N = A.todense()
        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                if N[i,j] != N[j,i]: print(i,j)
        print('None i,j means Symmetrix')
    def svd_embedding(self, L, i,eg,evec):
        u, s, vh = np.linalg.svd(L, full_matrices=True)
        title = self.dataset_name
        np.save('spectral_emed/%s_%d_u_neg'% (title,i),u)
        np.save('spectral_emed/%s_%d_s_neg'% (title,i),s)
        np.save('spectral_emed/%s_%d_vh_neg'% (title,i),vh)
        np.save('spectral_emed/%s_%d_eg'% (title,i),eg)
        np.save('spectral_emed/%s_%d_evec'% (title,i),evec)
        print(s)
    def eigen_cal(self,affinity_list, newlen, selected_idxs):
        eigen_list = []
        print('--Matrix size: %s '%newlen)
        for bigi in range(len(affinity_list)):
            second = affinity_list[bigi]

            A = sparse.csr_matrix((newlen, newlen), dtype=np.float64)
            A+= second
            diag_list = np.asarray(np.sum(A.todense(),axis=0)[0])
            print('Start point:', diag_list.shape[0] - np.count_nonzero(diag_list))
        for bigi in range(len(affinity_list)):
            second = affinity_list[bigi]
            if bigi not in selected_idxs: continue
            A = sparse.csr_matrix((newlen, newlen), dtype=np.float64)
            A+= second
            # self.check_symmtrix(A)
            #print('NAN',np.isnan(A.todense()).any())
            print('================Calculating eigenvalues for graph number %s =================' %(bigi))
            # Change here if necessary, calculate k eigenvalues
            #print('NAN', np.count_nonzero(np.isnan(fractional_matrix_power((np.diag(np.asarray(np.sum(A.todense(),axis=0))[0])),-0.5))))
            #print('NAN',np.count_nonzero(np.isnan(np.dot(fractional_matrix_power((np.diag(np.asarray(np.sum(A.todense(),axis=0))[0])),-0.5),A.todense()))))
            #return 0
            #L = np.identity(len(np.asarray(np.sum(A.todense(),axis=0))[0])) - np.dot(np.dot(fractional_matrix_power((np.diag(np.asarray(np.sum(A.todense(),axis=0))[0])),-0.5),A.todense()),fractional_matrix_power((np.diag(np.asarray(np.sum(A.todense(),axis=0))[0])),-0.5))

            L = self.normalized_Laplacian(np.asarray(np.sum(A.todense(),axis=0)[0])\
                                          ,A.todense())
            #u, s, vh = np.linalg.svd(L, full_matrices=True)
            #print(s)
            print('NAN',np.isnan(L).any())
            print('Inf',np.isinf(L).any())
            diag_list = np.asarray(np.sum(A.todense(),axis=0)[0])
            print('Start point:', diag_list.shape[0] - np.count_nonzero(diag_list))
            egval_1,egvec_1 = np.linalg.eig(L)
            # self.svd_embedding(L,bigi,egval_1,egvec_1)
            eigen_list.append(egval_1)
        return eigen_list

'''
    Modify the following function to adjust the plot
'''
def label_and_plot( eigen_list, label_li, title,selected_idxs):
    f_list = []
    for eli in eigen_list:
        f_list.append(np.sort(eli))
    np.save('eigenlist/%s'% (title),f_list)
#         f_list = np.load('eigenlist/%s.npy' %(title))
    # fig = plt.figure()
    # fig.set_size_inches(21, 10.5)
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    # cm = plt.get_cmap('gist_rainbow')
    index = 0
    # ax.set_color_cycle([cm(1.*i/len(selected_idxs)) for i in range(len(selected_idxs))])
    shape_li = ['k.', 'ko', 'bx', 'b+', 'r^', 'r*', 'gd', 'bo','ro']
    for i in selected_idxs:
#         if i == 2: continue
        # plt.plot([num for num in range(len(f_list[i]))],f_list[i], shape_li[index], label = label_li[index])
        plt.plot([num for num in range(len(f_list[index]))],f_list[index], shape_li[index], label = label_li[index])
        index+=1
        # plt.plot(f_list[i], label = label_li[i])
    #     plt.plot(f_list[i][0:4000], label = label_li[i])

    # ax.legend(bbox_to_anchor=(1.05,1), loc='best', shadow=True, \
    #           fancybox=True, borderaxespad=0.)
    #leg.get_frame().set_alpha(0.5)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlim(0,len(f_list[0]))
    plt.ylim(0,2)
    plt.xlabel('K', fontsize=25)
    plt.ylabel('Eigenvalues', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=12, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig('fig/%s.png' % (title), format='png', dpi=200, bbox_inches='tight')
    #plt.show()
    #plt.clf()
    print('Plot saved to fig folder')


def main():
    if sys.argv[1] == 'dblp':
        class_id_mapping = {'a': 0, 'p': 1, 'v': 2} # for dblp
        input_schema = {'a':{}, 'p' : {}, 'v': {}}
        selected_class = 'a' # for dblp
    elif sys.argv[1] == 'uspatent':
        class_id_mapping = {'c': 0, 'a': 1, 'i': 2, 'p': 3} # for uspatent
        input_schema = {'a':{}, 'c' : {}, 'i': {}, 'p' : {}}
        selected_class = 'p' # for uspatent
    elif sys.argv[1] == 'imdb':
        class_id_mapping =  {"a": 0, "m": 1, "d": 2, "u": 3} # for imdb
        input_schema = {'a':{}, 'm' : {}, 'd': {}, 'u' : {}}
        selected_class = 'm' # for imdb
    elif sys.argv[1] == 'yelp':
        class_id_mapping =  {"b": 0, "c": 1, "u": 2} # for yelp
        input_schema = {'b':{}, 'c' : {}, 'u': {}}
        selected_class = 'b' # for yelp
    else:
        raise Exception('Undefined Dataset!',sys.argv[1])
#    label_li = ['PIPI (F1-Macro: 0.074, F1-Micro:0.209)','PAPA (F1-Macro: 0.111, F1-Micro:0.239)', 'PAP (F1-Macro: 0.133, F1-Micro:0.249)','PIP (F1-Macro: 0.053, F1-Micro:0.189)','PP (F1-Macro: 0.064, F1-Micro:0.2)']
    #label_li = [' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',' (F1-Macro: 0., Instance #: )',]
    dataset_name = sys.argv[1]
    # dataset_name = 'dblp'
    selected_indexs = [0,1,2,3,4,6,7,8]
    # selected_indexs = [0,1,2,3,4,5]
    eig_analy = eig_analysis(dataset_name, selected_class, class_id_mapping,\
                            # input_schema,[0,1,2,3,4,6,7,8])
                            input_schema,selected_indexs)
                            # input_schema,[0,1,2,3,4,6,7,8,9,10,11,12])
    eig_analy.graph()
    rid_li, cid_li, data_li, affinity_list,instancecount = eig_analy.load_subgraph()
    # label_li = ['MAM','MDM','MUM','M(UD)M','M(AD)M','M(UA)M','M(UAD)M','MUMAM'\
    #             'MUMDM','MAMDM','MUMDMAM']
    # label_li = ['PIPI(Circle)','PAPA(Circle)','PAP','PIP','PP','PIPA(Circle)','PPP','PPA(Circle)','PPI(Circle)']
    # label_li = ['BUB','B(CU)B','B(UCU)B','B(CUU)B','B(UU)B']
    label_li = ['A(PP)A',\
                'APA',\
                'APVPA',\
                'APPA',\
                'APAPA',\
                'APPPA',\
                'APPAPPA',\
                'PAPAP'\
                # 'PP (F1-Macro: 0.246, Instance #:%d )' % instancecount[9],\
                # 'PPP (F1-Macro: 0.246, Instance #:%d )' % instancecount[10],\
                # 'PVP (F1-Macro: 0.246, Instance #:%d )' % instancecount[11],\
                # 'PAP (F1-Macro: 0.107, Instance #:%d )' % instancecount[12]
                ]
    affinity_mat,newlen = eig_analy.filter_small_component(rid_li, cid_li, data_li, affinity_list)
    # The next line is an function to further filter the affinity matrix
    # affinity_list = eig_analy.filter_API([affinity_list[q] for q in [i,j]])
    #########################################################
    eigvalues = eig_analy.eigen_cal(affinity_mat,newlen,selected_indexs)
    label_and_plot(eigvalues, label_li, '%s_start_point_weighted_1' % dataset_name,eig_analy.select_idx)

main()
