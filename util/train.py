import torch_geometric.utils as U


def train(
    params: typing.Dict
) -> torch.nn.Module:
  """
    This function trains a node classification model and returns the trained model object.
  """
  # set device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # load dataset
  data = dataset.data
  data = data.to(device)

  # Update parameters
  params["n_classes"] = dataset.num_classes # number of target classes
  params["input_dim"] = dataset.num_features # size of input features

  # Set a model
  if params['model_name'] == 'GCN':
      print('using GCN')
      model = GCN(
        params["input_dim"], 
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
        ).to(device)
  elif params['model_name'] == 'SkipGCN':
      model = SkipGCN(
        params["input_dim"], 
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
      ).to(device)
  elif params['model_name'] == 'JumpKnowGCN':
      model = JumpKnowGCN(
        params["input_dim"], 
        params["hid_dim"],
        params["n_classes"],
        params["n_layers"]
      ).to(device)
  
  elif params['model_name'] == 'GAT':
      model = GAT(
      params["input_dim"], 
      params["hid_dim"],
      params["n_classes"],
      params["n_layers"],
      in_heads = params["in_heads"],
      concat = params["concat"] 
    ).to(device)
  elif params['model_name'] == 'MyGAT':
      model = MyGAT(
      params["input_dim"], 
      params["hid_dim"],
      params["n_classes"],
      params["n_layers"]
    ).to(device)
  else:
      raise NotImplementedError
  model.param_init()
  print(model)
  
  opt = torch.optim.Adam(model.parameters(),lr=params['lr'],weight_decay=params['weight_decay'])
  loss_fn = nn.CrossEntropyLoss()
  train_losses, train_accuracies, test_accuracies = [],[],[]
  patience = params['max_patience']
  prev_acc = 0
  p_count = 0
  best_val = 0
  best_test = 0
  for ep in range(params['epochs']):
    if p_count==patience:
      print("early stoping at",ep)
      break
          
    opt.zero_grad()
    loss = 0
    x = data.x
    y_true = data.y
    mask = data.train_mask
    y_pred = model(x,data.edge_index)
    loss = loss_fn(y_pred[mask],y_true[mask])
    train_losses.append(loss)
    _, predicted = torch.max(y_pred.data, 1)
    train_acc = sum(predicted[mask]==y_true[mask]).item()/len(y_true[mask])
    if ep%10==0:
      print(train_acc,'at ep',ep)
    # print(train_acc,prev_acc)
    
    train_accuracies.append(train_acc)
    test_acc = evaluate(model, data, data.test_mask)
    best_test = max(best_test,test_acc)
    val_acc = evaluate(model, data, data.val_mask)
    if val_acc<=prev_acc:
      p_count+=1
    else:
      prev_acc = val_acc
      p_count = 0
    best_val = max(best_val,val_acc)
    # print(test_acc,val_acc)
    test_accuracies.append(test_acc)
      # if (ep%8)==0:
        # print('loss:',train_losses, 'train_acc',train_accuracies, 'test_acc',test_accuracies)

    
    loss.backward()
    opt.step()
  print(best_val,best_test)
  return model,best_test
  
  
