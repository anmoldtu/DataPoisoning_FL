from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.utils.client_utils import log_client_data_statistics
from client import Client
import torch
import time
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import datetime
import pickle

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    with torch.no_grad():
        initial_global_parameters = parameters_to_vector(clients[0].get_net_parameters()).detach()
    
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    
    with torch.no_grad():
        parameter_updates = [(parameters_to_vector(clients[client_idx].get_net_parameters()) - initial_global_parameters) for client_idx in random_workers]
    
    param_size = len(parameters_to_vector(clients[0].get_net_parameters()))
    print(param_size)

    new_nn_param_updates = average_nn_parameters(parameter_updates)

    
    print(new_nn_param_updates)
    if args.noise > 0:
            print("Noise!!")
            new_nn_param_updates.add_(torch.normal(mean=0, std=args.noise*args.clip, size=(param_size,)))
    print(new_nn_param_updates)
    new_nn_params = initial_global_parameters + new_nn_param_updates

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        with torch.no_grad():
            vector_to_parameters(new_nn_params, client.get_net_parameters())
        dictpy = client.get_nn_parameters()
        client.update_nn_parameters(dictpy)

    return clients[0].test(), random_workers

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results), worker_selection

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx, net, LR):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_default_args(net, LR)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.set_noise_and_clip(KWARGS["NOISE"], KWARGS["CLIP"],)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())

    #Uncomment when you need to save distributed dataset
#     logger.info("Creating New Distribution!")
#     with open(args.distributed_train_dataset_path, "wb") as f:
#         pickle.dump(distributed_train_dataset, f)

    if(args.distributed_train_data_exist == True):
        logger.info("Distribution Exists! Loading ...")
        with open(args.distributed_train_dataset_path, 'rb') as pickle_file:
            distributed_train_dataset = pickle.load(pickle_file)
  
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)


    logger.info("Data Distribution before Poisoning:")
    class_labels = list(set(distributed_train_dataset[0][1]))
#     print(class_labels)
    all_client_classes_dist = log_client_data_statistics(logger, class_labels, distributed_train_dataset)
    
    print(all_client_classes_dist)
    
    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    
#     poisoned_workers = [24, 44, 23, 30, 27]
    
    if(args.get_num_poisoned_workers() > 0):
        if(net != 'fashion-mnist'):
            poisoned_workers = [1, 21, 47, 29, 11]
        else:
            poisoned_workers = [24, 44, 23, 30, 27]
    
#     if(net != 'fashion-mnist'):
#         if(idx < 13):
#             # Less Instances
#             poisoned_workers = [12, 14,  5, 37, 10] 
#         else:
#             # More Instances
#             poisoned_workers =[ 2, 18, 31, 38, 40]
#     else:
#         if(idx < 17):
#             # Less Instances
#             poisoned_workers = [17, 48, 19, 36, 16]
#         else:
#             # More Instances
#             poisoned_workers =[26,  7, 21, 18, 22]
    
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)
    
    start_time = time.time()
    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    end_time = time.time()
    
    train_time = end_time - start_time
    logger.info("Training Time is {}".format(str(datetime.timedelta(seconds = train_time))))
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
