import torch
from torch.utils.data import DataLoader
from model import BClassifier, PanDataset
from torchmetrics.classification import BinaryAveragePrecision
from preprocess import PreProcessor
from sklearn.metrics import confusion_matrix

def run(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations):
    p = PreProcessor(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations)
    data, pids, days, labels = p.pickle_processing()
   
    input_shape = data.shape[1]
    model = BClassifier(input_shape, 128)
    metric = BinaryAveragePrecision(thresholds=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PanDataset(days, pids, data,labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # Move the model to the selected device
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Set the number of training epochs
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (x, y) in enumerate(dataloader):
            # Move the input features and labels to the selected device
            x = x.to(device).float()
            y = y.to(device)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs.squeeze(), y.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_preds.append(outputs.detach().cpu())
            all_labels.append(y.cpu())  # Store the true labels for the current batch

        pred = torch.cat(all_preds, dim=0)
        true_labels = torch.cat(all_labels, dim=0)
        bap = metric(pred.squeeze(), true_labels)
        print(f"BAP: {bap}")
        # Convert the model predictions to binary (0 or 1)
        binary_pred = (pred.squeeze() > 0.5).int()

        # Calculate the confusion matrix
        cm = confusion_matrix(true_labels.numpy(), binary_pred.numpy())
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        print("Confusion matrix:\n", cm)