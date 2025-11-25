def calculate_accuracy(predictions, labels):
    """Calculate the accuracy of predictions against labels."""
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def calculate_loss(predictions, labels, loss_function):
    """Calculate the loss using the specified loss function."""
    return loss_function(predictions, labels)