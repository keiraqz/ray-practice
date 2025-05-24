import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np

import ray

### Helper functions ###

def get_data_loader():
    """
    Safely downloads data. Returns training/validation set dataloader.
    """
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ) # Converts MNIST images to tensors and normalizes them to mean = 0.1307, std = 0.3081.

    # Torch's DataLoader is not threadsafe, use Filelock
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, transform=mnist_transforms
            ),
            batch_size=128,
            shuffle=True,
        )
    return train_loader, test_loader

def evaluate(model, test_loader):
    """
    Evaluates the accuracy of the model on a validation dataset.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Evaluates only the first ~1024 test samples for speed.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


### Model ###

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3) # 1 input channel (grayscale), 3 output channels, 3x3 kernel.
        self.fc = nn.Linear(192, 10) # fc layer takes 192 features (based on flattening conv output) → 10 classes.

    def forward(self, x): 
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_weigths(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


### Parameter Server ###

@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradients_zip).sum(axis=0) for gradients_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weigths()

    def get_weigths(self):
        return self.model.get_weigths()

@ray.remote
class DataWorker(object):
    def __init__(self):
        self.model = ConvNet()
        self.data_iterator = iter(get_data_loader()[0])

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        return self.model.get_gradients()

if __name__ == "__main__":
    iterations = 200
    num_workers = 2

    model = ConvNet()
    test_loader = get_data_loader()[1]

    """ Synchronous Parameter Server Training
    1.  All workers receive the same current_weights.
    2.  Each worker computes gradients based on its batch.
    3.  The parameter server waits for all gradients to come back.
    4.  Once all are ready, it averages the gradients and updates the model.
    5.  New weights are broadcast again.

    Pros:
    •   Deterministic updates — all gradients are used in each update step.
    •   Good for convergence stability.

    Cons:
    •   Slowest worker bottlenecks the whole system (straggler problem).
    •   Not ideal in highly variable or large-scale environments.
    """
    print("Running synchronous parameter server training.")

    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-2)
    workers = [DataWorker.remote() for i in range(num_workers)]

    current_weights = ps.get_weigths.remote()
    for i in range(iterations):
        gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
        # Calculate update after all gradients are available.
        current_weights = ps.apply_gradients.remote(*gradients)

        if i % 10 == 0:
            # Evaluate the current model after every 10 steps.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))
    # Clean up Ray resources and processes before the next example.
    ray.shutdown()


    """ Asynchronous Parameter Server Training
    1.  Workers start computing gradients using the current weights.
    2.  As soon as any worker finishes, the server:
        •   Immediately applies that gradient.
        •   Sends updated weights back to that worker.
    3.  This continues asynchronously, with no waiting for other workers.

    Pros:
    •   Fastest possible throughput — no blocking.
    •   Efficient for large-scale systems where some nodes may be slow or fail.
    •   Great for real-time and fault-tolerant learning.

    Cons:
    •   Gradients may be stale (based on older weights).
    •   Higher variance in convergence.
    •   Requires more tuning to be stable.

    """
    print("Running Asynchronous Parameter Server Training.")

    ray.init(ignore_reinit_error=True)
    ps = ParameterServer.remote(1e-2)
    workers = [DataWorker.remote() for i in range(num_workers)]

    current_weights = ps.get_weigths.remote()

    gradients = {}
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    for i in range(iterations * num_workers):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))

