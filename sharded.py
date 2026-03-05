import numpy as np
from hashlib import sha256
import importlib
import json

def sizeOfShard(container, shard):
    '''
    Returns the size (in number of points) of the shard.
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    return shards[shard].shape[0]

def getShardHash(container, shard, until=None):
    '''
    Returns a hash of the indices of the points in the shard lower than until (separated by :).
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)

    if until == None:
        until = shards[shard].shape[0]
    indices = shards[shard][:until]
    string_of_indices = ':'.join(indices.astype(str))
    return sha256(string_of_indices.encode()).hexdigest()

def fetchShardBatch(container, shard, batch_size, dataset, class_name=0, offset=0, until=None):
    '''
    Generator returning batches of points in the shard that are not in the requests
    with specified batch_size from the specified dataset
    optionnally located between offset and until (slicing).
    '''
    shards = np.load('containers/{}/splitfile.npy'.format(container), allow_pickle=True)
    
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))
    if until == None or until > shards[shard].shape[0]:
        until = shards[shard].shape[0]

    limit = offset
    while limit <= until - batch_size:
        limit += batch_size
        indices = shards[shard][limit-batch_size:limit]
        yield dataloader.load(indices, label=class_name)
    if limit < until:
        indices = shards[shard][limit:until]
        yield dataloader.load(indices, label=class_name)

def fetchValBatch(dataset, batch_size):
    '''
    Generator returning batches of points from the specified val dataset
    with specified batch_size.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_val'] - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), category='val')
    if limit < datasetfile['nb_val']:
        yield dataloader.load(np.arange(limit, datasetfile['nb_val']), category='val')

def fetchTestBatch(dataset, batch_size):
    '''
    Generator returning batches of points from the specified test dataset
    with specified batch_size.
    '''
    with open(dataset) as f:
        datasetfile = json.loads(f.read())
    dataloader = importlib.import_module('.'.join(dataset.split('/')[:-1] + [datasetfile['dataloader']]))

    limit = 0
    while limit <= datasetfile['nb_test'] - batch_size:
        limit += batch_size
        yield dataloader.load(np.arange(limit - batch_size, limit), category='test')
    if limit < datasetfile['nb_test']:
        yield dataloader.load(np.arange(limit, datasetfile['nb_test']), category='test')
