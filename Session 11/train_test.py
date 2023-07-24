from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

train_losses = []
test_losses = []
train_acc = []
test_accuracies = []
train_loss = []
train_accuracies = []

def get_correct_count(prediction, labels):
    return prediction.argmax(dim=1).eq(labels).sum().item()


def get_incorrect_preds(prediction, labels):
    prediction = prediction.argmax(dim=1)
    indices = prediction.ne(labels).nonzero().reshape(-1).tolist()
    return indices, prediction[indices].tolist(), labels[indices].tolist()

# Train Function
def train(model, device, lr_scheduler, criterion, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Predict
        pred = model(data)
        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item() * len(data)
        # Backpropagation
        loss.backward()
        optimizer.step()
        correct += get_correct_count(pred, target)
        processed += len(data)
        pbar.set_description(desc= f'Batch_id={batch_idx}')
        lr_scheduler.step()

    train_acc = 100 * correct / processed
    train_loss /= processed
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    print(f"\nTrain Average Loss: {train_losses[-1]:0.4f}")
    print(f"Train Accuracy: {train_accuracies[-1]:0.2f}%")
    lrs = lr_scheduler.get_last_lr()
    print(f"Maximum Learning Rate: ", max(lrs))

# Test Function
def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    processed = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            test_loss += criterion(pred, target).item() * len(data)
            correct += get_correct_count(pred, target)
            processed += len(data)
            # Convert the predictions and target to class indices
            pred_classes = pred.argmax(dim=1)
            target_classes = target
            # Check for misclassified images
            misclassified_mask = ~pred_classes.eq(target_classes)
            misclassified_images.extend(data[misclassified_mask])
            misclassified_labels.extend(target_classes[misclassified_mask])
            misclassified_predictions.extend(pred_classes[misclassified_mask])

    test_acc = 100 * correct / processed
    test_loss /= processed
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)
    print(f"Test Average loss: {test_loss:0.4f}")
    print(f"Test Accuracy: {test_acc:0.2f}%")
    return misclassified_images[:10], misclassified_labels[:10], misclassified_predictions[:10]
