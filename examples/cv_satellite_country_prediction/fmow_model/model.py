
from torchvision import transforms, datasets, models
from torch import nn, optim
from torch.utils.data import DataLoader
import torch


MODEL_FNAME = "fmow-fra-rus.tmsd"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30


def new_model(classifier_training_only=False):
    model = models.squeezenet1_1(pretrained=True)

    # optionally turn off training
    if classifier_training_only:
        for param in model.parameters():
            param.requires_grad = False

    # update model to output two classes
    num_labels = 2
    model.classifier[1] = nn.Conv2d(512, num_labels, kernel_size=(1, 1), stride=(1, 1))

    # ensure training is enabled on the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def save_model(model: nn.Module):
    torch.save(model.state_dict(), MODEL_FNAME)


def load_model(fname=MODEL_FNAME) -> nn.Module:
    model = new_model()
    model.load_state_dict(torch.load(fname))
    return model


def train_model(model: nn.Module, train: DataLoader, test: DataLoader, epochs: int = DEFAULT_EPOCHS,
                disable_cuda: bool = False):
    device = torch.device("cuda") if (not disable_cuda and torch.cuda.is_available()) else torch.device("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters())

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        accuracy = 0

        # Training the model
        model.train()
        counter = 0
        for inputs, labels in train:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear optimizers
            optimizer.zero_grad()

            # Forward pass
            output = model.forward(inputs)

            # Loss
            loss = criterion(output, labels)

            # Calculate gradients (backpropogation)
            loss.backward()

            # Adjust parameters based on gradients
            optimizer.step()

            # Add the loss to the training set's running loss
            train_loss += loss.item() * inputs.size(0)

            # Print the progress of our training
            counter += 1
            print(f"{counter} / {len(train)}", end="\r")
        print("\n", end="\r")

        # Evaluating the model
        model.eval()
        counter = 0
        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in test:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                output = model.forward(inputs)

                # Calculate Loss
                valloss = criterion(output, labels)

                # Add loss to the validation set's running loss
                val_loss += valloss.item() * inputs.size(0)

                top_p, top_class = output.topk(1, dim=1)

                # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)

                # Calculate the mean (get the accuracy for this batch) and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print the progress of our evaluation
                counter += 1
                print(f"{counter} / {len(test)}", end="\r")

        if epoch % 10 == 4:
            print("saving intermediate")
            torch.save(model.state_dict(), f"fmow-fra-rus-epoch_{epoch + 1}.tmsd")

        # Get the average loss for the entire epoch
        train_avgloss = train_loss / len(train.dataset)
        val_avgloss = val_loss / len(test.dataset)

        # Print out the epoch information
        print('Accuracy: ', accuracy / len(test))
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(epoch + 1,
                                                                                     train_avgloss, val_avgloss))


def create_datasets(category: str, batch_size: int = DEFAULT_BATCH_SIZE):
    train_dir = f"fmow-data/{category}/train"
    test_dir = f"fmow-data/{category}/test"

    # TODO: add quantization?
    preprocess = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.ImageFolder(train_dir, transform=preprocess)
    test_set = datasets.ImageFolder(test_dir, transform=preprocess)

    train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_set, batch_size=1, shuffle=False)
    return train, test


def train_on_category(model: nn.Module, category: str, batch_size: int = DEFAULT_BATCH_SIZE,
                      epochs: int = DEFAULT_EPOCHS, disable_cuda: bool = False):
    train, test = create_datasets(category, batch_size)
    train_model(model, train, test, epochs, disable_cuda)
