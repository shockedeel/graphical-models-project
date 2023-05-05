import torch
from torch.utils.data import DataLoader
from model import BClassifier, PanDataset
from torchmetrics.classification import BinaryAveragePrecision
from preprocess import PreProcessor
from sklearn.metrics import confusion_matrix
import pandas as pd

def run(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations):
    p = PreProcessor(va_activity_loc_assign, va_activity_locations, va_disease_outcome_target, va_disease_outcome_training, va_household, va_person, va_population_network, va_residence_locations)
    data, pids, days, labels = p.pickle_processing()
    
    # path = 'C:/Users/kolbe/OneDrive/Desktop/Work/Filtered/bias_outcome/'
    # va_activity_loc_assignf = pd.read_csv(f'{path}va_activity_location_assignment.csv.gz', compression='gzip').iloc[:,1:]
    # va_activity_locationsf = pd.read_csv(f'{path}va_activity_locations.csv.gz', compression='gzip').iloc[:,1:]
    # va_disease_outcome_targetf = pd.read_csv(f'{path}va_disease_outcome_target.csv.gz', compression='gzip').iloc[:,1:]
    # va_disease_outcome_trainingf = pd.read_csv(f'{path}va_disease_outcome_training.csv.gz', compression='gzip')
    # va_householdf = pd.read_csv(f'{path}va_household.csv.gz', compression = 'gzip').iloc[:,1:]
    # va_personf = pd.read_csv(f'{path}va_person.csv.gz', compression='gzip')
    # va_population_networkf = pd.read_csv(f'{path}va_population_network.csv.gz', compression='gzip')
    # va_residence_locationsf = pd.read_csv(f'{path}va_residence_locations.csv.gz', compression='gzip').iloc[:,1:]
    # p = PreProcessor(va_activity_loc_assignf, va_activity_locationsf, va_disease_outcome_targetf, va_disease_outcome_trainingf, va_householdf, va_personf, va_population_networkf, va_residence_locationsf)
    # data_ub, pids_ub, days_ub, labels_ub = p.filtered_data_process()
    # ub_dataset = PanDataset(days_ub, pids_ub, data_ub, labels_ub)
    # ub_dataloader = DataLoader(ub_dataset, batch_size=64, shuffle=False)
    input_shape = data.shape[1]
    model = BClassifier(input_shape, 128)
    metric = BinaryAveragePrecision(thresholds=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PanDataset(days, pids, data,labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    # Move the model to the selected device
    model.to(device)
    negative_class_weight = 1.0
    positive_class_weight = 2.0  # Assign a higher weight to the positive (minority) class

# Create a tensor with the class weights
    class_weights = torch.tensor([negative_class_weight, positive_class_weight], dtype=torch.float).to(device)
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
    
    

    return model