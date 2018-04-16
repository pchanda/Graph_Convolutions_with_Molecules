import deepchem as dc
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.feat.mol_graphs import ConvMol
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, L2Loss, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
import numpy as np
import tensorflow as tf
import os


num_epochs = 80
batch_size = 100
pad_batches = True

tg = TensorGraph(batch_size=batch_size,learning_rate=0.0005,use_queue=False)
tox21_tasks = ['cLogP','cLogS']

def read_data(fname):
  # read the tox 21 data for a single phenotype
  dataset_file = os.path.join('./DW_props/', fname)
  featurizer='GraphConv'
  if featurizer == 'ECFP':
    featurizer = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = dc.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = dc.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = dc.feat.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = dc.feat.AdjacencyFingerprint(max_n_atoms=150, max_valence=6)

  loader = dc.data.CSVLoader(tasks=tox21_tasks, smiles_field="Smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)
  # datasets has 4 arrays : X (7831 x 1024), y (7831 x 12), w (7831 x 12) and ids (7831 x )

  # Initialize transformers
  transformer = dc.trans.NormalizationTransformer(transform_w=True, dataset=dataset)
  print("About to transform data")
  dataset = transformer.transform(dataset)

  splitter = dc.splits.splitters.RandomSplitter()
  trainset,testset = splitter.train_test_split(dataset,frac_train=0.8)
  
  return trainset,testset

print('About to train models')

# placeholder for a feature vector of length 75 for each atom
atom_features = Feature(shape=(None, 75))  
# an indexing convenience that makes it easy to locate atoms from all molecules with a given degree
degree_slice = Feature(shape=(None, 2), dtype=tf.int32) 
# placeholder that determines the membership of atoms in molecules (atom i belongs to molecule membership[i])
membership = Feature(shape=(None,), dtype=tf.int32)
# list that contains adjacency lists grouped by atom degree
deg_adjs = []
for i in range(0, 10 + 1):
   deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32) # placeholder for adj list of all nodes with i neighbors
   deg_adjs.append(deg_adj)

gc1 = GraphConv(64, activation_fn=tf.nn.relu, in_layers=[atom_features, degree_slice, membership]+deg_adjs )
batch_norm1 = BatchNorm(in_layers=[gc1])
gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)

gc2 = GraphConv(64,activation_fn=tf.nn.relu,in_layers=[gp1, degree_slice, membership] + deg_adjs)
batch_norm2 = BatchNorm(in_layers=[gc2])
gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)

dense = Dense(out_channels=512, activation_fn=tf.nn.relu, in_layers=[gp2])
batch_norm3 = BatchNorm(in_layers=[dense])
readout = GraphGather( batch_size=batch_size, activation_fn=tf.nn.tanh, in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)

costs = []
labels = []
for task in range(len(tox21_tasks)):
    regression = Dense( out_channels=1, activation_fn=None, in_layers=[readout])
    tg.add_output(regression)
    label = Label(shape=(None, 1))
    labels.append(label)
    cost = L2Loss(in_layers=[label, regression])
    costs.append(cost)

all_cost = Stack(in_layers=costs, axis=1)
weights = Weights(shape=(None, len(tox21_tasks)))
loss = WeightedError(in_layers=[all_cost, weights])
tg.set_loss(loss)


def data_generator(dataset, epochs=1, predict=False, pad_batches=True):
  for epoch in range(epochs):
    if not predict:
        print('Starting epoch %i' % epoch)
    data_iterator_batch = dataset.iterbatches(batch_size, pad_batches=pad_batches, deterministic=True)
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(data_iterator_batch):
      d = {} #sort of feed_dict
      for index, label in enumerate(labels):
        d[label] = np.expand_dims(y_b[:, index],1)
      d[weights] = w_b
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      d[atom_features] = multiConvMol.get_atom_features()
      d[degree_slice] = multiConvMol.deg_slice
      d[membership] = multiConvMol.membership
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
      yield d


def reshape_y_pred(y_true, y_pred):
    """
    TensorGraph.Predict returns a list of arrays, one for each output
    We also have to remove the padding on the last batch
    Metrics taks results of shape (samples, n_task, prob_of_class)
    """
    n_samples = len(y_true)
    retval = np.stack(y_pred, axis=1)
    return retval[:n_samples]

#Now, we can train the model using TensorGraph.fit_generator(generator) which will use the generator we’ve defined to train the model.
# Epochs set to 1 to render tutorials online.
# Set epochs=10 for better results.
train_dataset,test_dataset = read_data("Sample_input_10000.csv")
tg.fit_generator(data_generator(train_dataset, epochs=num_epochs))

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean,  mode = "regression")

print("Evaluating on train data")
train_predictions = tg.predict_on_generator(data_generator(train_dataset, predict=True))
train_predictions = reshape_y_pred(train_dataset.y, train_predictions)
train_scores = metric.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
print("Train Correlation Score: %f" % train_scores)

print("Evaluating on test data")
test_predictions = tg.predict_on_generator(data_generator(test_dataset, predict=True))
test_predictions = reshape_y_pred(test_dataset.y, test_predictions)
test_scores = metric.compute_metric(test_dataset.y, test_predictions, test_dataset.w)
print("Test Correlation Score: %f" % test_scores)

