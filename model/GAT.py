#@title class GAT 
from torch_geometric.nn import GATConv
class GAT(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3,
      in_heads: int = 8,
      concat:bool = True):
    super(GAT, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout_ratio
    """
    ## ------ Begin Solution ------ ##
    
    self.layer_num = n_layers
    self.d_ratio = dropout_ratio
    self.in_heads = in_heads
    self.out_head = 1
    self.attention_list = []
    layer_list =[]
    if n_layers==0:
      layer_list = [torch_geometric.nn.Linear(input_dim,n_classes)]
    else:
      # in_dim = input_dim
      # out_dim = hid_dim
      if n_layers==1:
        layer_list=[GATConv(input_dim,n_classes,heads=self.out_head)]
      else:
        layer_list=[GATConv(input_dim,hid_dim,heads=self.in_heads,concat = concat)]
        for i in range(n_layers-2):
          if concat:
            layer_list += [GATConv(hid_dim*in_heads,hid_dim,heads=in_heads,concat = concat)]
          else:
            layer_list += [GATConv(hid_dim,hid_dim,heads=in_heads,concat = concat)]
          # self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head)
          # self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False, #  heads=self.out_head)
          # layer_list += [GATConv(hid_dim*selfin_heads,hid_dim*self.in_heads,heads=self.in_heads, concat=concat)]
          # in_dim = hid_dim*self.in_head
        if concat:
          layer_list+=[GATConv(hid_dim*in_heads,n_classes,heads=self.out_head)]
        else:    
          layer_list+=[GATConv(hid_dim,n_classes,heads=self.out_head)]
    self.list = nn.ModuleList(layer_list)
    

    

  def forward(self, X, A,return_attention_weights = None) -> torch.Tensor:
    
    
    if self.layer_num==0:
      layer = self.list[0]
      X = layer(X)
      X = F.relu(X)
      X = F.dropout(X,p = self.d_ratio)
      return X
    else:
      for id,layer in enumerate(self.list):
        if return_attention_weights:
          X,att = layer(X,A,return_attention_weights = True)
          self.attention_list.append(att[1])
          print(len(self.attention_list),end = " ")
        else:
          X = layer(X,A)
        if id!=len(self.list)-1:          
          X = F.relu(X)       
          X = F.dropout(X,p = self.d_ratio) 
      
      return X
    
    
    

  def generate_node_embeddings(self, X, A) -> torch.Tensor:
    

    with torch.no_grad():
      
      return self.forward(X,A)
      
    
  def generate_attention(self, X, A) -> torch.Tensor:
    
    self.attention_list = []
    with torch.no_grad():
      
      self.forward(X,A,return_attention_weights = True)
      return self.attention_list
  
  def param_init(self):
    
    def _reset_module_parameters(module):
      for layer in module.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
          for child_layer in layer.children():
            _reset_module_parameters(child_layer)

    _reset_module_parameters(self)
    
