import torch
from torchvision import datasets, transforms

from gln.model import GLN, Layer
from gln.one_vs_all import OneVsAll

INPUT_DIM = 28 ** 2
CLASS = 0
LOG_INTERVAL = 100


def main():
    model = OneVsAll(10, create_model)
    train_data = data_loader(True)

    train_correct = 0
    train_total = 0
    for t, (inputs, outputs) in enumerate(train_data):
        logits = model.forward_grad(inputs, outputs)
        preds = torch.argmax(logits, dim=-1)
        train_correct += torch.sum((preds == outputs).long()).item()
        train_total += inputs.shape[0]

        lr = min(100 / max(t, 1), 0.01)
        for p in model.parameters():
            if p.grad is not None:
                p.detach().sub_(lr * p.grad)
                p.grad.zero_()

        if t % LOG_INTERVAL == 0 and t > 0:
            print(
                f"train {t}: train_accuracy={(train_correct/train_total):02f} lr={lr}"
            )
            train_total = 0
            train_correct = 0

    correct = 0
    total = 0
    test_data = data_loader(False)
    for inputs, outputs in test_data:
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=-1)
        correct += torch.sum((preds == outputs).long()).item()
        total += inputs.shape[0]
    print(f"test accuracy: {(correct/total):02f}")


def create_model():
    return GLN(
        Layer(INPUT_DIM, INPUT_DIM, 128),
        Layer(INPUT_DIM, 128, 128),
        Layer(INPUT_DIM, 128, 1),
    )


def data_loader(train):
    mnist = datasets.MNIST(
        "mnist_data", train=train, download=True, transform=transforms.ToTensor()
    )
    for x, y in torch.utils.data.DataLoader(
        mnist, batch_size=(1 if train else 128), shuffle=True
    ):
        yield x.view(x.shape[0], -1), y


if __name__ == "__main__":
    main()
