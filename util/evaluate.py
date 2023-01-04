def evaluate(
    model,
    data,
    mask
):
    
    with torch.no_grad():
      x = data.x
      y_pred = model(x,data.edge_index)
      _, predicted = torch.max(y_pred.data, 1)
      
      y_true = data.y #[data.test_mask]
      # print('eval on ',len(y_true[mask]))
      acc = sum(predicted[mask]==y_true[mask]).item()/len(y_true[mask])
      # print(acc)
      return acc
    
    
