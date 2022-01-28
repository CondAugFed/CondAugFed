import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.strategy import FedAvg

def fit_config_fn(round):
    return {'round':round}

def eval_config_fn(round, num_rounds):
    return {'round':round, 'num_rounds':num_rounds}

class ModifiedFedAvg(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        num_rounds=None,
    ) -> None:

        super(ModifiedFedAvg, self).__init__(fraction_fit,
                                             fraction_eval,
                                             min_fit_clients,
                                             min_eval_clients,
                                             min_available_clients,
                                             eval_fn,
                                             on_fit_config_fn,
                                             on_evaluate_config_fn)
        
        if num_rounds is None:
            raise AttributeError('num_rounds should be given.')
        
        self.num_rounds = num_rounds

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd, self.num_rounds)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]