import boto3
from itertools import groupby
import pydicom
import pandas as pd
from collections import defaultdict
import pickle as pkl

def get_institutions(bucket, prefix):
      s3 = boto3.client('s3')
      pickle_file = "./uni_keys.pkl"
      try:
            uni_keys = pkl.load(open(pickle_file, 'rb'))
      except FileNotFoundError:
            print("Not found making list")
            paginator = s3.get_paginator('list_objects')
            pages = paginator.paginate(Bucket=bucket,Prefix=prefix)
            uni_keys = []
            uni_keys_suffix = []
            for page in pages:
                  conts = page['Contents']
                  for c in conts:
                        k = c['Key']
                        if k.endswith('.dcm'):
                              suffix = k.split("_")[-1]
                              if suffix not in uni_keys_suffix:
                                    uni_keys_suffix.append(suffix)
                                    uni_keys.append(k)
            pkl.dump(uni_keys, open(pickle_file, 'wb'))
      print(len(uni_keys))
      pickle_file = "./keys.pkl"
      try:
            keys = pkl.load(open(pickle_file, 'rb'))
      except FileNotFoundError:
            keys = []
            for i in uni_keys:
                  key_dict = {}
                  key_dict['key'] = i
                  key_dict['subject_id'] = i.split('/')[3]
                  key_dict['image_id'] = i.split('_')[-1].replace('.dcm','')
                  filename = '/tmp/outfile.dcm'
                  s3.download_file(bucket, i, filename)
                  ds = pydicom.dcmread(filename)
                  try:
                        inst = ds.InstitutionName
                        if inst == '':
                              inst = '<missing>'
                  except AttributeError:
                        inst = '<missing>'
                  key_dict['institution'] = inst
                  keys.append(key_dict)
            pkl.dump(keys, open(pickle_file, 'wb'))
      df = pd.DataFrame(keys)
      return df


def clustering2(df):
      # make a dict for each patno with all the institutions
      print(df)
      clusters = {}
      for i,r in df.iterrows():
            inst = r['institution']
            patno = r['subject_id']
            try:
                  found = clusters[patno]
                  found.append(inst)
                  clusters[patno] = found
            except KeyError:
                  clusters[patno] = [inst,]
      inst_lists = [v for k,v in clusters.items()]
      graph = {}
      for i in range(len(inst_lists)):
            pairs = []
            for j in inst_lists[i]:
                  x = j
                  for m in inst_lists[i]:
                        pairs.append((x,m))
            graph[i] = pairs

      old_graph = graph
      edges = {v for k, vs in old_graph.items() for v in vs}
      graph = defaultdict(set)

      for v1, v2 in edges:
          graph[v1].add(v2)
          graph[v2].add(v1)

      components = []
      for component in connected_components(graph):
          c = set(component)
          components.append([edge for edges in old_graph.values()
                                  for edge in edges
                                  if c.intersection(edge)])

      value_list = []
      for i in components:
            values = []
            for j in i:
                  values.append(j[0])
                  values.append(j[1])
            value_list.append(values)
      mapper = {}
      for v in value_list:
            mc = max(set(v), key = v.count)
            mapper[mc] = v

      old_new_map = {}
      for k,v in mapper.items():
            for j in v:
                  old_new_map[j] = k
      df['institution_clustered'] = df.apply(lambda x: old_new_map[x['institution']], axis=1)
      return df



def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)




if __name__ == "__main__":
      bucket = 'ppmi-image-data'
      prefix = 'NEW_PPMI/DCM_RAW/PPMI/'
      inst = get_institutions(bucket, prefix)
      df = clustering2(inst)
      #df = pd.merge(inst, sub_inst_map, on='subject_id')

      df.to_csv('./institution_map.csv', index=False)
