#@title class GCN 
class GCN(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      dropout_ratio: float = 0.3):
    super(GCN, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout_ratio
    """
    
    
    self.layer_num = n_layers
    self.d_ratio = dropout_ratio
    layer_list =[]
    if n_layers==0:
      layer_list = [torch_geometric.nn.Linear(input_dim,n_classes)]
    else:
      in_dim = input_dim
      out_dim = hid_dim
      for i in range(n_layers-1):
        layer_list += [GCNConv(in_dim,out_dim)]
        in_dim = hid_dim
      layer_list+=[GCNConv(in_dim,n_classes)]
      
    self.list = nn.ModuleList(layer_list)
    

    

  def forward(self, X, A) -> torch.Tensor:
    
    if self.layer_num==0:
      layer = self.list[0]
      X = layer(X)
      X = F.relu(X)
      X = F.dropout(X,p = self.d_ratio)
      return X
    else:
      for id,layer in enumerate(self.list):
        X = layer(X,A)
        if id!=len(self.list)-1:          
          X = F.relu(X)       
          X = F.dropout(X,p = self.d_ratio) 
      
      return X
    
    
    

  def generate_node_embeddings(self, X, A) -> torch.Tensor:
    

    with torch.no_grad():
      
      return self.forward(X,A)
      
    
  
  def param_init(self):
    
    def _reset_module_parameters(module):
      for layer in module.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
          for child_layer in layer.children():
            _reset_module_parameters(child_layer)

    _reset_module_parameters(self)
    
