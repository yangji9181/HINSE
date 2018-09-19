import numpy as np
import pandas as pd
import sys
import os

class Dblp2node_link:
    def __init__(self):
        self.input_paper_author = {}
        self.input_author_paper = {}
        self.input_author_name = {}
        self.input_name_author = {}
        self.venue = {}
        self.selected_author = set()
        self.selected_papaer = set()
        self.selected_paper_venue = {}
        self.reference_list = []
        self.sanity_check = set()

    def read_author(self,filename):
        i = 0
        with open(filename) as input_author:
            for line in input_author:
                if i == 0:
                    i+=1
                    print(line)
                    continue
                toks = line.replace('\n','').replace(' ','_').split(',')
                i+=1
                if toks[1] not in self.input_paper_author:
                    self.input_paper_author[toks[1]] = set()
                if toks[0] not in self.input_author_paper:
                    self.input_author_paper[toks[0]] = set()
                self.input_paper_author[toks[1]].add(toks[0])
                self.input_author_paper[toks[0]].add(toks[1])

    def read_author_name(self,filename):
        i = 0
        with open(filename) as input_person:
            for line in input_person:
                if i == 0:
                    i+=1
                    continue
                toks = line.replace('\n','').split(',')
                i+=1
                self.input_author_name[toks[0]] = toks[1]
                self.input_name_author[toks[1].replace(' ','_')] = toks[0]

    def read_paper_venue_year(self, filename, selected_venue):
        paper_table = pd.read_csv(filename)
        abbreviation_flag = True
        if len(selected_venue) < 1:
            abbreviation_flag = False
            venue_file = filename.replace('paper','venue')
            with open(venue_file) as venues:
                for line in venues:
                    venue_name = line.replace('\n','')
                    self.venue[venue_name] = set()
        print(len(self.venue), abbreviation_flag)
        if abbreviation_flag:
            self.venue['ECML_PKDD'] =set()
            self.venue['SIGMOD_PODS'] =set()
            for venue in selected_venue:
                self.venue[venue] = set()
                selected_paper_table = paper_table[paper_table['venue'].str.contains(venue)]
                if venue == 'KDD':
                    selected_paper_table = selected_paper_table[~selected_paper_table['venue'].str.contains('PAKDD')]
                    selected_paper_table = selected_paper_table[~selected_paper_table['venue'].str.contains('PKDD')]
                elif venue == 'SDM':
                    selected_paper_table = selected_paper_table[~selected_paper_table['venue'].str.contains('WSDM')]
                for index, row in selected_paper_table.iterrows():
                    if row['id'] in self.selected_paper_venue:
                        if venue == 'PKDD' and self.selected_paper_venue[row['id']] == 'ECML':
                            self.selected_paper_venue[row['id']] = 'ECML_PKDD'
                            continue
                        elif venue == 'PODS' and self.selected_paper_venue[row['id']] == 'SIGMOD':
                            self.selected_paper_venue[row['id']] = 'SIGMOD_PODS'
                            continue
                        print(self.selected_paper_venue[row['id']], row['venue'],venue)
                    self.selected_paper_venue[row['id']] = venue
                    self.selected_papaer.add(row['id'])
        else:
            for venue in self.venue:
                selected_paper_table = paper_table[paper_table['venue'] == venue]
#                 print(selected_paper_table.shape)
                for index, row in selected_paper_table.iterrows():
                    self.selected_paper_venue[row['id']] = venue
                    self.selected_papaer.add(row['id'])

    def back_filter_read_ref(self,filename):
        '''
        Selecting authors for filtered paper and read P-P link
        '''
        for paper in self.selected_papaer:
            if str(paper) in self.input_paper_author:
                self.selected_author |= self.input_paper_author[str(paper)]
            else:
                self.sanity_check.add(str(paper))
        i = 0
        with open(filename) as input_reference:
            for line in input_reference:
                if i == 0 :
                    i+=1
                    continue
                toks = line.replace('\n','').split(',')
                if int(toks[0]) in self.selected_papaer and int(toks[1]) in self.selected_papaer:
                    self.reference_list.append(str(toks[0]) + '\t' + str(toks[1]))
                i+=1
        print(len(self.reference_list))

    def filter_by_author(self, author_file):
        temp_selected_author = set()
        with open(author_file) as author_list:
            for line in author_list:
                if line.replace('\n','') in self.input_name_author:
                    temp_selected_author.add(self.input_name_author[line.replace('\n','')])
        temp_selected_paper = set()
        missing_count = 0
        for author in temp_selected_author:
            if author in self.input_author_paper:
                temp_selected_paper |= self.input_author_paper[author]
            else:
                missing_count+=1
        print(missing_count, 'in groud truth not in dataset')
        temp_reference_list = []
        for token in self.reference_list:
            toks = token.split('\t')
            if str(toks[0]) in temp_selected_paper and str(toks[1]) in temp_selected_paper:
                temp_reference_list.append(token)
        self.selected_papaer = temp_selected_paper
        self.selected_author = temp_selected_author
        self.reference_list = temp_reference_list

    def output_to_file(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        node_list = []
        link_list = list(self.reference_list)
        print('p-p edge count',len(link_list))
        refined_author = set()
        author_count = 0
        for paper in self.selected_papaer:
            node_list.append(str(paper) + '\tp')
            if str(paper) not in self.input_paper_author:
#                 print(paper)
                continue
            for author_id in self.input_paper_author[str(paper)]:
                if author_id in self.selected_author:
                    link_list.append(str(self.input_author_name[author_id].replace(' ','_'))+'\t'+str(paper))
                    link_list.append(str(paper)+'\t'+str(self.input_author_name[author_id].replace(' ','_')))
        print('p-p+p-a edge count',len(link_list))
        for author in self.selected_author:
            if author not in self.input_author_name: continue
            refined_author.add(str(self.input_author_name[author].replace(' ','_')))
        for final_author in refined_author:
            node_list.append(final_author + '\ta')
            author_count+=1
        for venue in self.venue:
            node_list.append(str(venue.replace(' ','_')) + '\tv')
        for paper in self.selected_paper_venue:
            link_list.append(str(paper) +'\t'+ str(self.selected_paper_venue[paper]))
            link_list.append(str(self.selected_paper_venue[paper]) +'\t'+ str(paper))
        print('node count:',len(node_list),';','author count:',author_count,';','edge count:',len(link_list))
        with open(save_dir+'node.dat','w') as node_output:
            for node in node_list:
                node_output.write(node+'\n')
        with open(save_dir+'link.dat','w') as link_output:
            for link in link_list:
                link_output.write(link+'\n')
    def sanity_check_func(self,filename):
        i = 0
        with open(filename) as input_author:
            for line in input_author:
                if i == 0:
                    i+=1
                    continue
                toks = line.replace('\n','').split(',')
                i+=1
                if toks[1] in self.sanity_check:
                    print('WRONG',toks)
# numwalks = int(sys.argv[1])
"""The following data is extracted from Aminer database. Please follow their instructions to
select the year range. After selecting year, you could algo select venues as you want.
"""
authorpath = "../filtered_data/author-1995-2014.csv"
refpath = "../filtered_data/refs-1995-2014.csv"
paperpath = "../filtered_data/paper-1995-2014.csv"
personpath = "../filtered_data/person-1995-2014.csv"
groundtruth = "/shared/data/mengqu2/data_dblp/eval/clus_dblp/vocab-label.txt"
# selected_venue = ['AAAI','CVPR','ECML','IJCAI','SIGMOD','VLDB','PODS','EDBT','ICDE','ICDM','KDD','PAKDD','PKDD','ECIR','SIGIR','WSDM','WWW','CIKM']
# selected_venue = ['ICDM','KDD','PAKDD','PKDD','WSDM','WWW','CIKM']
# selected_venue = ['KDD','ECML','PKDD']
selected_venue = []

save_dir = 'data/dblp/'
def main():
    dblp2data =Dblp2node_link()
    dblp2data.read_author(authorpath)
    dblp2data.read_author_name(personpath)
    dblp2data.read_paper_venue_year(paperpath,selected_venue)
    dblp2data.back_filter_read_ref(refpath)
    print('Begin filtering authors')
    dblp2data.filter_by_author(groundtruth)
    dblp2data.output_to_file(save_dir)
    dblp2data.sanity_check_func(authorpath)

main()
