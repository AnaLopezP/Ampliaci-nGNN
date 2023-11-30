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

#Para la creación de los modelos
from torch.nn import Linear, Dropout
from torch_geometric.nn import SAGEConv, GATv2Conv, GCNConv
import torch.nn.functional as F


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


#Implementación de la arquitectura GraphSAGE
class GraphSAGE(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.sage1 = SAGEConv(dim_in, dim_h)
    self.sage2 = SAGEConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index).relu()
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    return F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
      total_loss = 0
      acc = 0
      val_loss = 0
      val_acc = 0

      # Entrenamiento de los lotes
      for batch in train_loader:
        optimizer.zero_grad()
        out = self(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        total_loss += loss
        acc += accuracy(out[batch.train_mask].argmax(dim=1),
                         batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # Validación
        val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
        val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                             batch.y[batch.val_mask])

      # Imprimimos las métricas cada 10 épocas
      if(epoch % 10 == 0):
          print(f'Epoch {epoch:>3} | Train Loss: {total_loss/len(train_loader):.3f} '
                f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                f'{val_loss/len(train_loader):.2f} | Val Acc: '
                f'{val_acc/len(train_loader)*100:.2f}%')


#Lo mismo pero con GAT para comparar
class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=heads)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer
    self.train()
    for epoch in range(epochs+1):
        # Entrenamiento
        optimizer.zero_grad()
        out = self(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1),
                       data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validación
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])

        # Imprimimos las métricas cada 10 épocas
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                  f' {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')

#Lo mismo pero con GCN para comparar
class GCN(torch.nn.Module):
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(h, edge_index).relu()
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    return F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer
    self.train()

    for epoch in range(epochs+1):
        # Entrenamiento
        optimizer.zero_grad()
        out = self(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1),
                       data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validación
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1),
                           data.y[data.val_mask])

        # Imprimimos las métricas cada 10 épocas
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc:'
                  f' {acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')


#Calculamos la precisión
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

#Evaluamos los modelos y imprimimos su precisión
@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

#Creamos GraphSAGE
graphsage = GraphSAGE(dataset.num_features, 64, dataset.num_classes)
print(graphsage)

#Entrenamiento
graphsage.fit(data, 200)

#Evaluación
print(f'Precisión de GraphSAGE: {test(graphsage, data)*100:.2f}%')

#Lo mismo pero con GCN
gcn = GCN(dataset.num_features, 64, dataset.num_classes)
print(gcn)
gcn.fit(data, 200)
print(f'Precisión de GCN: {test(gcn, data)*100:.2f}%')

#lo mismo pero con GAT
gat = GAT(dataset.num_features, 64, dataset.num_classes)
print(gat)
gat.fit(data, 200)
print(f'Precisión de GAT: {test(gat, data)*100:.2f}%')