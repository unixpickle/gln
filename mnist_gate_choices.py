import math

from train_mnist import create_model, data_loader


def main():
    model = create_model()
    train_data = data_loader(True)

    counters = None
    for inputs, _ in train_data:
        choices = model.layers[0].gate_choices(inputs).view(-1).numpy()
        if counters is None:
            counters = [dict() for _ in choices]
        for counter, choice in zip(counters, choices):
            counter[choice] = counter.get(choice, 0) + 1
        ent = sum(entropy(counter) for counter in counters) / len(counters)
        print("average of %f nats" % ent)


def entropy(counter):
    total = sum(counter.values())
    log_sum = 0.0
    for c in counter.values():
        log_sum += math.log(c / total) * c / total
    return -log_sum


if __name__ == "__main__":
    main()
