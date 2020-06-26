import argparse
from functools import partial

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms

from gln.model import GLN, Layer
from gln.one_vs_all import OneVsAll

INPUT_DIM = 28 ** 2
CLASS = 0
LOG_INTERVAL = 100

DEVICE = torch.device("cuda" if torch.cuda.device_count() else "cpu")


def main():
    args = arg_parser().parse_args()

    model = OneVsAll(
        10,
        partial(
            create_model,
            half_spaces=args.half_spaces,
            epsilon=args.epsilon,
            weight_clip=args.weight_clip,
        ),
    )
    model.to(DEVICE)

    opt = optim.SGD(model.parameters(), lr=args.lr)
    schedule = LambdaLR(opt, lambda t: min(10000 / (t + 1), 1.0))

    train_correct = 0
    train_total = 0
    train_data = data_loader(True, deterministic=args.deterministic)
    for t, (inputs, outputs) in enumerate(train_data):
        opt.zero_grad()

        logits = model.forward_grad(inputs, outputs)
        preds = torch.argmax(logits, dim=-1)
        train_correct += torch.sum((preds == outputs).long()).item()
        train_total += inputs.shape[0]

        opt.step()
        schedule.step()
        model.clip_weights()

        if t % LOG_INTERVAL == 0 and t > 0:
            print(f"train {t}: train_accuracy={(train_correct/train_total):02f}")
            train_total = 0
            train_correct = 0

    print(f"train accuracy: {compute_accuracy(model, data_loader(True, batch=128))}")
    print(f"test accuracy: {compute_accuracy(model, data_loader(False, batch=128))}")


def compute_accuracy(model, dataset):
    correct = 0
    total = 0
    for inputs, outputs in dataset:
        with torch.no_grad():
            preds = torch.argmax(model(inputs), dim=-1)
        correct += torch.sum((preds == outputs).long()).item()
        total += inputs.shape[0]
    return correct / total


def create_model(**kwargs):
    return GLN(
        Layer(INPUT_DIM, INPUT_DIM, 128, **kwargs),
        Layer(INPUT_DIM, 128, 128, **kwargs),
        Layer(INPUT_DIM, 128, 1, **kwargs),
    )


def data_loader(train, batch=1, deterministic=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    mnist = datasets.MNIST(
        "mnist_data", train=train, download=True, transform=transform
    )
    for x, y in torch.utils.data.DataLoader(
        mnist, batch_size=batch, shuffle=not deterministic
    ):
        yield x.view(x.shape[0], -1).to(DEVICE), y.to(DEVICE)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.01, help="initial learning rate", type=float)
    parser.add_argument(
        "--half-spaces", default=4, help="number of half-space gates", type=int
    )
    parser.add_argument("--epsilon", default=1e-4, help="sigmoid clip", type=float)
    parser.add_argument("--weight-clip", default=10, help="weight clip", type=float)
    parser.add_argument("--deterministic", action="store_true")
    return parser


if __name__ == "__main__":
    main()
