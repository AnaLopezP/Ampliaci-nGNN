import torch
#para la visualizacion
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import Planetoid

#para el muestreo de vecinos más adelante
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx

#Para las gráficas de los grados de nodos
from torch_geometric.utils import degree
from collections import Counter

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 12})

#vamos a utilizar el conjunto de datos de PubMed
#cargam
# os el conjunto de datos e imprimimos información sobre el grafo

dataset = Planetoid(root = '.', name = "Pubmed")
data = dataset[0]

#info del dataset
print(f'dataset: {dataset}:')
print('-------------------')
print(f'Número de grafos: {len(dataset)}')
print(f'Número de nodos: {data.x.shape[0]}')
print(f'Número de características: {dataset.num_features}')
print(f'Número de clases: {dataset.num_classes}')

#info del grafo
print(f'\nGraph:')
print('------')
print(f'nodos de entrenamiento: {sum(data.train_mask).item()}')
print(f'nodos de evaluación: {sum(data.val_mask).item()}')
print(f'nodos para test: {sum(data.test_mask).item()}')
print(f'Grafo direccional: {data.is_directed()}')
print(f'Grafo con nodos aislados: {data.has_isolated_nodes()}')
print(f'Grafo con lazos: {data.has_self_loops()}')

#Muestreo de vecinos
#primero creamos loa lotes
train_loader = NeighborLoader(
    data,
    num_neighbors=[5, 10],
    batch_size=16,
    input_nodes = data.train_mask)

#imprimimos todos los subgrafos
for i, subgrafo in enumerate(train_loader):
    print(f'Subgrafo {i}: {subgrafo}')
    
#visualizamos los subgrafos
fig = plt.figure(figsize=(16, 16))
for idx, (subdata, pos) in enumerate(zip(train_loader, [221, 222, 233, 224])):
    G = to_networkx(subdata, to_undirected=True)
    ax = fig.add_subplot(pos)
    ax.set_title(f'Subgrafo {idx}')
    plt.axis('off')
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size = 200,
        node_color = subdata.y,
        cmap = 'cool',
        font_size = 10
        )

plt.show()

def plot_degree(data):
    # cojemos la lista de grados por cada nodo
    degrees = degree(data.edge_index[0]).numpy()

    # contamos el numero de nodos por grado
    numbers = Counter(degrees)

    # mostramos la figura
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlabel('Node degree')
    ax.set_ylabel('Number of nodes')
    plt.bar(numbers.keys(),
            numbers.values(),
            color='#0A047A')

# mostramos los grados de los nodos del grafo completo
plot_degree(data)

# mostramos los grados de los nodos de los subgrafos
plot_degree(subdata)