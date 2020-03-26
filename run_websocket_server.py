import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms

import syft as sy
from syft.workers import websocket_server
from random import sample 
import pickle



    
def start_websocket_server_worker(id, host, port, hook, verbose, dataset, training=True):
    """Helper function for spinning up a websocket server and setting up the local datasets."""
    
    server = websocket_server.WebsocketServerWorker(
        id=id, host=host, port=port, hook=hook, verbose=verbose
    )
    
    
    # --------------- Setup toy data ----------------#
    
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    if(dataset == "fashionmnist"):
        toy_dataset = datasets.FashionMNIST('./data', train = training, download=True,
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),)

        dataset_key = "fashionmnist"
    
    if(dataset == "mnist"):
        toy_dataset = datasets.MNIST(root="./data", train=training, download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                ]),)    
        dataset_key = "mnist"
        
    if(dataset == "cifar10"):
        toy_dataset = datasets.CIFAR10(root="./data", train=training, download=True,
                   transform= transform)
        dataset_key = "cifar10"

        
    
    if training:
        with open("./split/%d" % int(id), "rb") as fp:   # Unpickling
            split = pickle.load(fp)      
        indices = np.isin([i for i in range(len(toy_dataset.data))], split)
        logger.info("Number of true indices for client %s is %s : ", id, indices.sum())
        
        
        if(dataset == "mnist" or dataset == "fashionmnist"):
            selected_data = (
            torch.native_masked_select(toy_dataset.data.transpose(0, 2), torch.tensor(indices))
            .view(28, 28, -1)
            .transpose(2, 0)
            )
        
        if(dataset == "cifar10"):
            selected_data = (
                torch.native_masked_select(torch.tensor(toy_dataset.data.transpose((3,1,2,0))), torch.tensor(indices))
                .view(3, 32, 32, -1)
                .transpose(3, 0)
            )

        logger.info("after selection: %s", selected_data.shape)
        selected_targets = torch.native_masked_select(torch.tensor(toy_dataset.targets), torch.tensor(indices))
        dataset = sy.BaseDataset(
            data=selected_data, targets=selected_targets, transform=toy_dataset.transform
        )
        key = dataset_key
    else:
        dataset = sy.BaseDataset(
            data=torch.tensor(toy_dataset.data),
            targets=torch.tensor(toy_dataset.targets),
            transform=toy_dataset.transform,
        )
        key = dataset_key+ "_testing"

        
    server.add_dataset(dataset, key=key)
    count = [0] * 10
    logger.info(
        "CIFAR dataset (%s set), available numbers on %s: ", "train" if training else "test", id
    )
    for i in range(10):
        count[i] = (dataset.targets == i).sum().item()
        logger.info("      %s: %s", i, count[i])
    
    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("len(datasets): %s", len(server.datasets[key]))

    server.start()
    return server


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)

    # Parse args
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument("--host", type=str, default="localhost", help="host for the connection")
    parser.add_argument(
        "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )
    parser.add_argument("--dataset", help="Dataset held by users")

    args = parser.parse_args()
    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        dataset=args.dataset,
        training=not args.testing,
    )
    