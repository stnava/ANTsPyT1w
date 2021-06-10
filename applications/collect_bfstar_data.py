import ia_batch_utils as batch
import pandas as pd

def qa(procid, filename):
    df = pd.DataFrame()
    for i in procid:
        new = batch.collect_data(i, '')
        df = pd.concat([df,new])
    df.to_csv(f's3://invicro-data-outputs/dynamoqa/{filename}-stacked.csv')
    df = batch.pivot_data(df)
    df.to_csv(f's3://invicro-data-outputs/dynamoqa/{filename}-pivoted.csv')
    print(filename)
    print(df.shape)
    print("******")
    return df

def join_all(dfs, how):
    df = dfs[0]
    for d in dfs[1:]:
        merge = pd.merge(df, d, on='originalimage', how=how, suffixes=('', "_y"))
        cols = [i for i in merge.columns if i.endswith('_y')]
        merge.drop(cols, axis=1, inplace=True)
        print(merge.shape)
        df = merge

    return df

bf_star_or_old = qa(['A7DA'], 'bf_star_or_old')
bf_star_or_new = qa(['687D'], 'bf_star_or_new')
#bf_star_sr = qa(['2BF9'], 'bf_star_sr')
#rbp = qa(['A134'], 'rbp')
#deephipp = qa(['4D50'], 'deephip')
#hemi_sr = qa(['7CB4'], 'hemi_sr')
#deepl = qa(['E3C8'], 'deepl')
#deepr = qa(['275B'], 'deepr')
meta = pd.read_csv('s3://eisai-basalforebrainsuperres2/metadata/full_metadata_20210208.csv')
meta['originalimage'] = meta['filename']
#data = [bf_star_or, bf_star_sr, rbp, deephipp, hemi_sr, deepl, deepr]
data = [bf_star_or_old, bf_star_or_new]
vols = join_all(data, "left")
df = join_all([meta,vols], "right")
df.to_csv('s3://eisai-basalforebrainsuperres2/reproducibility_test_bf_star_on_or.csv')
