from glob import glob
import os 
import sys
import pickle
from collections import defaultdict
TOP = "data/ebm_nlp_2_00/"
DOC_PKL = 'docs.pkl'
def fname_to_pmid(fname):
  pmid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
  return pmid

def build_data():

  print('Reading documents...')

  docs = {}
  doc_fnames = glob('%s/documents/*.tokens' %(TOP))

  for i, fname in enumerate(doc_fnames):
    pmid = os.path.basename(fname).split('.')[0]

    tokens = open(fname).read().split()
    tags = open(fname.replace('tokens', 'pos')).read().split()
    docs[pmid] = {}
    docs[pmid]['tokens'] = tokens
    docs[pmid]['pos'] = tags

    if (i//100 != (i-1)//100):
      sys.stdout.write('\r\tprocessed %04d / %04d docs' %(i, len(doc_fnames)))
      sys.stdout.flush()

  with open(DOC_PKL, 'wb') as fout:
    print('\nWriting doc data to %s' %DOC_PKL)
    pickle.dump(docs, fout)

  return docs

def preprocess_data():
    ebm_nlp = 'data/ebm_nlp_2_00/'

    id_to_tokens = {}
    id_to_pos = {}
    PIO = ['participants', 'interventions', 'outcomes']
    PHASES = ['starting_spans', 'hierarchical_labels']

    token_fnames = glob('%s/documents/*.tokens' %ebm_nlp) # Loop through all the documents that have tokens
    for fname in token_fnames:
      pmid = fname_to_pmid(fname)
      tokens = open(fname).read().split()
      tags   = open(fname.replace('tokens', 'pos')).read().split()
      id_to_tokens[pmid] = tokens
      id_to_pos[pmid] = tags

    batch_to_labels = {}
    for phase in PHASES:
      batch_to_labels[phase] = {}
      for pio in PIO:
        batch_to_labels[phase][pio] = {}
        print('Reading files for %s %s' %(phase, pio))
        for fdir in ['train', 'test/gold']:
          batch = fdir.split('/')[-1]
          batch_to_labels[phase][pio][batch] = dict()
          ann_fnames = glob('%s/annotations/aggregated/%s/%s/%s/*.ann' %(ebm_nlp, phase, pio, fdir))
          for fname in ann_fnames:
            pmid = fname_to_pmid(fname)
            batch_to_labels[phase][pio][batch][pmid] = open(fname).read().split()

    batch_groups = [('p1_all', ['starting_spans'], ['participants', 'interventions', 'outcomes']),
                    ('p2_p', ['hierarchical_labels'], ['participants']),
                    ('p2_i', ['hierarchical_labels'], ['interventions']),
                    ('p2_o', ['hierarchical_labels'], ['outcomes'])]
    for group_name, phases, pio in batch_groups:

      id_to_labels_list = defaultdict(list)
      batch_to_ids = defaultdict(set)
      for phase in phases:
        for e in pio:
          print('Collecting anns from %s %s' %(phase, e))
          for batch, batch_labels in batch_to_labels[phase][e].items():
            print('\t%d ids for %s' %(len(batch_labels), batch))
            batch_to_ids[batch].update(batch_labels.keys())
            for pmid, labels in batch_labels.items():
              labels = ['%s_%s' %(l, e[0]) for l in labels]
              id_to_labels_list[pmid].append(labels)

      for batch, ids in batch_to_ids.items():
        print('Found %d ids for %s' %(len(ids), batch))

      train_ids = list(batch_to_ids['train'] - batch_to_ids['gold'])
      print('Using %d ids for train' %len(train_ids))

      dev_idx = int(len(train_ids) * 0.2)
      dev_ids, train_ids = set(train_ids[:dev_idx]), set(train_ids[dev_idx:])
      print('Split training set in to %d train, %d dev' %(len(train_ids), len(dev_ids)))
      batch_to_ids['train'] = train_ids
      batch_to_ids['dev'] = dev_ids

      for batch, ids in batch_to_ids.items():
        fout = open('data/%s_%s.txt' %(group_name, batch), 'w')
        for pmid in ids:
          fout.write('-DOCSTART- -X- O O\n\n')
          tokens = id_to_tokens[pmid]
          poss = id_to_pos[pmid]
          per_token_labels = zip(*id_to_labels_list[pmid])
          for i, (token, pos, labels) in enumerate(zip(tokens, poss, per_token_labels)):
            final_label = 'N'
            for l in labels:
              if l[0] != '0':
                final_label = l
            fout.write('%s %s %s\n' %(token, pos, final_label))
            if token == '.': fout.write('\n')


if __name__ == "__main__":
    # docs = None
    # if not os.path.isfile(DOC_PKL):
    #     print('Building data file...')
    #     docs = build_data()
    # # preprocess_data()
    # docs = pickle.load(open(DOC_PKL, 'rb'))
    # print(len(docs))
    # print(docs['43164'].keys()) # {'tokens': [],
    #                             #  'pos': []
    #                             # }
    preprocess_data()