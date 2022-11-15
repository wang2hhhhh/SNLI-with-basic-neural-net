import matplotlib.pyplot as plt



#you can define your own plotting function with this template
def plot_over_training(per_epoch_metrics, title):
  """Utility function to plot train/val accuracies and losses.

  @param per_epoch_metrics: a dictionary of lists, where each list represents a metric over the
      course of training.
  @param title_name: String to show on the plot title.
  """
  t = np.arange(0, len(per_epoch_metrics['train_accs']))
  train_acc = per_epoch_metrics['train_accs']
  val_acc = per_epoch_metrics['val_accs']
  train_loss = per_epoch_metrics['train_losses']
  val_loss = per_epoch_metrics['val_losses']

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('epochs')
  ax1.set_ylabel('acc', color=color)
  ax1.plot(t, train_acc, color=color, linewidth=1, label = 'train_acc')
  ax1.plot(t, val_acc, color=color, linestyle='dashed', linewidth=1, label = 'val_acc')
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.legend(loc='upper left')
  ax2 = ax1.twinx() 

  color = 'tab:blue'
  ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
  ax2.plot(t, train_loss, color=color, linewidth=1, label = 'train_loss')
  ax2.plot(t, val_loss, color=color, linestyle='dashed', linewidth=1, label = 'val_loss')
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.legend(loc='lower right')
  fig.tight_layout() 
  plt.title(title)
  plt.show()



def print_exmaples(pred_dataloader, model):
    device = 'cuda'
    i,j = 0,0
    for hypo, len_h, prem, len_p, label in pred_dataloader:
        hypo.to(device)
        len_h.to(device) 
        prem.to(device)
        len_p.to(device)
        label.to(device)
    out = model(hypo, len_h, prem, len_p)
    preds = out.argmax(axis = 1)
    if (preds == label) and (i<3):
        print('right')
        print('prem ', [id2token[i] for i in prem.tolist()[0] if i !=0])  
        print('hypo ', [id2token[i] for i in hypo.tolist()[0] if i !=0])
        print('pred', preds)
        print('label', label)
        i+=1
    if (preds != label) and (j<3):
        print('wrong')
        print('prem ', [id2token[i] for i in prem.tolist()[0] if i !=0])
        print('hypo ', [id2token[i] for i in hypo.tolist()[0] if i !=0])
        print('pred', preds)
        print('label', label)
        j+=1