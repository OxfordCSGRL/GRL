device =  "cuda" if torch.cuda.is_available() else "cpu"
def visualize_attention(model: GAT):
    """
    Args:
      model: model object for generating features
    
    Return:
      pd.DataFrame: A data frame that has 'dimension 1', 'dimension 2', and 'labels' as a column
    """
    
    rep = model.generate_attention(data.x.to(device),data.edge_index)
    rep = rep[data.val_mask]
    
    rep = rep.cpu().detach().numpy()
    
    embedding = TSNE(n_components=2, learning_rate='auto',
                  init='random').fit_transform(rep)
    return embedding
  
if __name__=="__main__":
  device =  "cuda" if torch.cuda.is_available() else "cpu"
  demo = m10(data.x.to(device),data.edge_index, return_attention_weights = True)
