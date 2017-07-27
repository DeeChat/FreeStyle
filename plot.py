import matplotlib.pyplot as plt


def read_log(path):
    x = []
    y_acc = []
    y_loss = []
    count = 1
    with open(path) as f:
        for line in f:
            if line.startswith('| epoch'):
                segments = line.split('|')
                acc_seg = segments[-1].strip()
                loss_seg = segments[-3].strip()
                acc = float(acc_seg[-4:])
                loss = float(loss_seg[-4:])
                y_acc.append(acc)
                y_loss.append(loss)
                x.append(count)
                count += 1
    return x, y_acc, y_loss


def show(x, y, metric, title):
    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    paths = ['D:\\logs-s.txt', 'D:\\logs-l.txt']
    for path in paths:
        x, y_acc, y_loss = read_log(path)
        show(x, y_acc, 'Accuracy', path)
        show(x, y_loss, 'Loss', path)
