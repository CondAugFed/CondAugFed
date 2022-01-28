import argparse
import flwr as fl
from flwr.server.strategy import FedAvg
from src.federated.strategy.fedavg import ModifiedFedAvg, fit_config_fn, eval_config_fn

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--server_address', type=str, default='localhost:8085')
    parser.add_argument('--num_rounds', type=int, default=150)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    args = parser.parse_args()

    # This strategy means that using all given clients for training and evaluation.
    # Weight aggregation is done by FedAvg algorithm.
    strategy = ModifiedFedAvg(fraction_fit=args.sample_ratio,
                            fraction_eval=1.0,
                            min_fit_clients=1,
                            min_eval_clients=args.num_clients,
                            min_available_clients=args.num_clients,
                            on_fit_config_fn=fit_config_fn,
                            on_evaluate_config_fn=eval_config_fn,
                            num_rounds=args.num_rounds)

    fl.server.start_server(args.server_address, 
                           config={"num_rounds": args.num_rounds},
                           strategy=strategy)

if __name__ == '__main__':
    main()