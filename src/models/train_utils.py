from tqdm.notebook import tqdm_notebook
import torch


def train(model, train_dataloader, validation_dataloader, args, criterion, optimizer, loss_dict):

  train_losses = [] # record training loss after every epoch
  val_losses = [] # record validation loss after every epoch
  train_accs = [] # record training accuracy after every epoch
  val_accs = [] # record validation accuracy after every epoch

  train_losses_batch = [] # record training loss after every batch
  train_accs_batch = [] # record training accuracy after every batch


  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.to(device)

  pbar1 = tqdm_notebook(range(args.num_epochs), leave = False)
  for epoch in pbar1:

    model.train()
    train_loss_epoch = []
    train_accu_epoch = []

    pbar2 = tqdm_notebook(train_dataloader, leave = False)
    for hypo, len_h, prem, len_p, label in pbar2:

      hypo.to(device)
      len_h.to(device) 
      prem.to(device)
      len_p.to(device)
      label.to(device)

      out = model(hypo, len_h, prem, len_p)
      preds = out.argmax(axis = 1)
      batch_correct = (preds == label).float().sum()

      batch_loss = criterion(out, label)
      optimizer.zero_grad()
      batch_loss.backward()
      optimizer.step()
      
      #record train loss
      mean_batch_loss = batch_loss.item()/label.shape[0]
      loss_dict['train_loss_epoch'].append(mean_batch_loss)
      loss_dict['train_losses_batch'].append(mean_batch_loss)

      #record train accu
      batch_acc = batch_correct.item()/label.shape[0]
      loss_dict['train_accu_epoch'].append(batch_acc)
      loss_dict['train_accs_batch'].append(batch_acc)

      # pbar2.set_postfix({'mean_batch_loss': mean_batch_loss, 'batch_acc': batch_acc})

    mean_loss_epoch = sum(train_loss_epoch)/len(train_loss_epoch)
    mean_accu_epoch = sum(train_accu_epoch)/len(train_accu_epoch)

    loss_dict['train_losses'].append(mean_loss_epoch)
    loss_dict['train_accs'].append(mean_accu_epoch)

    val_loss =0
    val_correct = 0
    with torch.no_grad():
      for hypo, len_h, prem, len_p, label in validation_dataloader:
        hypo.to(device)
        len_h.to(device) 
        prem.to(device)
        len_p.to(device)
        label.to(device)
        out = model(hypo, len_h, prem, len_p)
        preds = out.argmax(axis = 1)
        batch_correct = (preds == label).float().sum()
        batch_loss = criterion(out, label)

        val_loss += batch_loss.item()
        val_correct += batch_correct

    mean_val_loss = val_loss/len(validation_dataloader.dataset)
    mean_val_accu = val_correct/len(validation_dataloader.dataset)

    loss_dict['val_losses'].append(mean_val_loss)
    loss_dict['val_accs'].append(mean_val_accu)

  pbar1.set_postfix({'mean_val_loss':mean_val_loss, 'mean_val_accu': mean_val_accu, 'mean_loss_epoch':mean_loss_epoch, 'mean_accu_epoch':mean_accu_epoch})

  return loss_dict, model 