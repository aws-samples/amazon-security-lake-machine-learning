from collections.abc import Iterable
import pandas as pd
import boto3
import os
import matplotlib.pyplot as plt


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def chunker(seq, size):
    """Return `seq` in chunks of size `size`"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def multiply_by_2(num):
    return num*2

def set_pandas_colwidth(w): pd.set_option('display.max_colwidth', w)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

def plot_date_dist(series, title=None, figsize=(15, 4), by='day', xlabel=None, ylabel=None):
    xlabel = 'Date' if xlabel is None else xlabel
    ylabel = f'Number Records per {by}' if ylabel is None else ylabel
    assert by in ['day', 'month'], f'invalid value passed for day parameter ({by})'
    columns = ['year', 'month', 'day']
    if by == 'day':
        gb = [series.dt.year, series.dt.month, series.dt.day]
    elif by == 'month':
        gb = [series.dt.year, series.dt.month]
    counts = series.groupby(gb).size().to_frame('num_recs')
    dateparts = list(counts.index.values)
    if by == 'month': dateparts = list(map(lambda x: list(x) + [1], dateparts))
    x = pd.to_datetime(pd.DataFrame(dateparts, columns=columns))
    y = counts['num_recs']
    fig, ax = plt.subplots(figsize=figsize)
    title = title if title is not None else f'{series.name} | Number of records per {by}'
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
