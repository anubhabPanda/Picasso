import config
from tqdm.notebook import tqdm
import torch

def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader))
    train_losses = []
    train_acc = []
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_fn(y_pred, target)
        train_losses.append(loss)
        
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdims=True)
        correct+= pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Batch ID ={batch_idx}, Loss={loss.item():0.3f}, Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100.*correct/processed)

        return model, train_losses, train_acc

def test(model, dataloader, loss_fn, device, n_misclassified):
    model.eval()
    pbar = tqdm(dataloader, total=len(dataloader))
    test_loss = 0
    test_losses = []
    test_accuracy = []
    misclassified_imgs = []

    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            batch_preds = model(data)
            test_loss += loss_fn(batch_preds, target, reduction='sum').item()
            preds = batch_preds.argmax(dim=1, keepdims=True)
            correct += preds.eq(target.view_as(preds)).sum().item()

            if len(misclassified_imgs) <=n_misclassified:
                incorrect_id = ~preds.eq(target.view_as(preds))

                if incorrect_id.sum().item() != 0:
                    for i in list(incorrect_id.cpu().numpy()):
                        misclassified_imgs.append({'img':data[i],
                                                    'pred':preds[i],
                                                    'target': target.view_as(preds)[i]})

    test_loss /= len(dataloader)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(dataloader),
        100. * correct / len(dataloader)))
    
    test_accuracy.append(100. * correct / len(dataloader))
    return test_loss, test_accuracy, misclassified_imgs

