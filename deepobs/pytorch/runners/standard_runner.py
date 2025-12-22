"""Module implementing StandardRunner for PyTorch."""

from __future__ import print_function

import argparse
import os
import json
import importlib
import torch
import numpy as np

from .. import config
from .. import testproblems
from . import runner_utils


class StandardRunner(object):
    """Provides functionality to run optimizers on DeepOBS testproblems including
    the logging of important performance metrics.

    This runner handles the complete training workflow:
    1. Parsing command-line arguments
    2. Setting up the test problem (dataset + model)
    3. Creating and configuring the optimizer
    4. Training loop with periodic evaluation
    5. Learning rate scheduling
    6. Metric logging and saving results to JSON

    Args:
        optimizer_class: PyTorch optimizer class (e.g., torch.optim.SGD).
        hyperparams (list): A list describing the optimizer's hyperparameters other
            than learning rate. Each entry of the list is a dictionary describing
            one of the hyperparameters. This dictionary is expected to have the
            following two fields:

              - hyperparams["name"] must contain the name of the parameter (i.e.,
                the exact name of the corresponding keyword argument to the
                optimizer class' init function).
              - hyperparams["type"] specifies the type of the parameter (e.g.,
                ``int``, ``float``, ``bool``).

            Optionally, the dictionary can have a third field indexed by the key
            "default", which specifies a default value for the hyperparameter.

    Example:
        >>> import torch.optim as optim
        >>> optimizer_class = optim.SGD
        >>> hyperparams = [
                {"name": "momentum", "type": float},
                {"name": "nesterov", "type": bool, "default": False}]
        >>> runner = StandardRunner(optimizer_class, hyperparams)
    """

    def __init__(self, optimizer_class, hyperparams):
        """Creates a new StandardRunner.

        Args:
            optimizer_class: PyTorch optimizer class (must inherit from
                torch.optim.Optimizer).
            hyperparams (list): A list describing the optimizer's hyperparameters
                other than learning rate. Each entry of the list is a dictionary
                describing one of the hyperparameters. This dictionary is expected
                to have the following two fields:
                  - hyperparams["name"] must contain the name of the parameter (i.e.,
                    the exact name of the corresponding keyword argument to the
                    optimizer class' init function).
                  - hyperparams["type"] specifies the type of the parameter (e.g.,
                    ``int``, ``float``, ``bool``).
                Optionally, the dictionary can have a third field indexed by the key
                "default", which specifies a default value for the hyperparameter.
        """
        self._optimizer_class = optimizer_class
        self._optimizer_name = optimizer_class.__name__
        self._hyperparams = hyperparams

    def run(self,
            testproblem=None,
            weight_decay=None,
            batch_size=None,
            num_epochs=None,
            learning_rate=None,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            random_seed=None,
            data_dir=None,
            output_dir=None,
            train_log_interval=None,
            print_train_iter=None,
            no_logs=None,
            **optimizer_hyperparams):
        """Runs a given optimizer on a DeepOBS testproblem.

        This method receives all relevant options to run the optimizer on a DeepOBS
        testproblem, including the hyperparameters of the optimizers, which can be
        passed as keyword arguments (based on the names provided via ``hyperparams``
        in the init function).

        Options which are *not* passed here will automatically be added as command
        line arguments. (Some of those will be required, others will have defaults;
        run the script with the ``--help`` flag to see a description of the command
        line interface.)

        Training statistics (train/test loss/accuracy) are collected and will be
        saved to a ``JSON`` output file, together with metadata.

        Args:
            testproblem (str): Name of a DeepOBS test problem.
            weight_decay (float): The weight decay factor to use.
            batch_size (int): The mini-batch size to use.
            num_epochs (int): The number of epochs to train.
            learning_rate (float): The learning rate to use. This will function as the
                base learning rate when implementing a schedule using
                ``lr_sched_epochs`` and ``lr_sched_factors`` (see below).
            lr_sched_epochs (list): A list of epoch numbers (positive integers) that
                mark learning rate changes. The base learning rate is passed via
                ``learning_rate`` and the factors by which to change are passed via
                ``lr_sched_factors``.
                Example: ``learning_rate=0.3``, ``lr_sched_epochs=[50, 100]``,
                ``lr_sched_factors=[0.1 0.01]`` will start with a learning rate of
                ``0.3``, then decrease to ``0.1*0.3=0.03`` after training for ``50``
                epochs, and decrease to ``0.01*0.3=0.003`` after training for ``100``
                epochs.
            lr_sched_factors (list): A list of factors (floats) by which to change the
                learning rate. The base learning rate has to be passed via
                ``learning_rate`` and the epochs at which to change the learning rate
                have to be passed via ``lr_sched_factors``.
                Example: ``learning_rate=0.3``, ``lr_sched_epochs=[50, 100]``,
                ``lr_sched_factors=[0.1 0.01]`` will start with a learning rate of
                ``0.3``, then decrease to ``0.1*0.3=0.03`` after training for ``50``
                epochs, and decrease to ``0.01*0.3=0.003`` after training for ``100``
                epochs.
            random_seed (int): Random seed to use. If unspecified, it defaults to
                ``42``.
            data_dir (str): Path to the DeepOBS data directory. If unspecified,
                DeepOBS uses its default `/data`.
            output_dir (str): Path to the output directory. Within this directory,
                subfolders for the testproblem and the optimizer are automatically
                created. If unspecified, defaults to 'results'.
            train_log_interval (int): Interval of steps at which to log training loss.
                If unspecified it defaults to ``10``.
            print_train_iter (bool): If ``True``, training loss is printed to screen.
                If unspecified it defaults to ``False``.
            no_logs (bool): If ``True`` no ``JSON`` files are created. If unspecified
                it defaults to ``False``.
            **optimizer_hyperparams: Keyword arguments for the hyperparameters of
                the optimizer. These are the ones specified in the ``hyperparams``
                dictionary passed to the ``__init__``.
        """
        # We will go through all the arguments, check whether they have been passed
        # to this function. If yes, we collect the (name, value) pairs in ``args``.
        # If not, we add corresponding command line arguments.
        args = {}
        parser = argparse.ArgumentParser(
            description="Run {0:s} on a DeepOBS test problem.".format(
                self._optimizer_name))

        if testproblem is None:
            parser.add_argument(
                "testproblem",
                help="""Name of the DeepOBS testproblem
          (e.g. 'mnist_mlp')""")
        else:
            args["testproblem"] = testproblem

        if weight_decay is None:
            parser.add_argument(
                "--weight_decay",
                "--wd",
                type=float,
                help="""Factor
          used for the weight_decay. If not given, the default weight decay for
          this model is used. Note that not all models use weight decay and this
          value will be ignored in such a case.""")
        else:
            args["weight_decay"] = weight_decay

        if batch_size is None:
            parser.add_argument(
                "--batch_size",
                "--bs",
                required=True,
                type=int,
                help="The batch size (positive integer).")
        else:
            args["batch_size"] = batch_size

        if num_epochs is None:
            parser.add_argument(
                "-N",
                "--num_epochs",
                required=True,
                type=int,
                help="Total number of training epochs.")
        else:
            args["num_epochs"] = num_epochs

        if learning_rate is None:
            parser.add_argument(
                "--learning_rate",
                "--lr",
                required=True,
                type=float,
                help=
                """Learning rate (positive float) to use. Can be used as the base
          of a learning rate schedule when used in conjunction with
          --lr_sched_epochs and --lr_sched_factors.""")
        else:
            args["learning_rate"] = learning_rate

        if lr_sched_epochs is None:
            parser.add_argument(
                "--lr_sched_epochs",
                nargs="+",
                type=int,
                help="""One or more epoch numbers (positive integers) that mark
          learning rate changes. The base learning rate has to be passed via
          '--learning_rate' and the factors by which to change have to be passed
          via '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")
        else:
            args["lr_sched_epochs"] = lr_sched_epochs

        if lr_sched_factors is None:
            parser.add_argument(
                "--lr_sched_factors",
                nargs="+",
                type=float,
                help=
                """One or more factors (floats) by which to change the learning
          rate. The base learning rate has to be passed via '--learning_rate' and
          the epochs at which to change the learning rate have to be passed via
          '--lr_sched_factors'. Example: '--lr 0.3 --lr_sched_epochs 50 100
          --lr_sched_factors 0.1 0.01' will start with a learning rate of 0.3,
          then decrease to 0.1*0.3=0.03 after training for 50 epochs, and
          decrease to 0.01*0.3=0.003' after training for 100 epochs.""")
        else:
            args["lr_sched_factors"] = lr_sched_factors

        if random_seed is None:
            parser.add_argument(
                "-r",
                "--random_seed",
                type=int,
                default=42,
                help="An integer to set as PyTorch's random seed.")
        else:
            args["random_seed"] = random_seed

        if data_dir is None:
            parser.add_argument(
                "--data_dir",
                help="""Path to the base data dir. If
      not specified, DeepOBS uses its default.""")
        else:
            args["data_dir"] = data_dir

        if output_dir is None:
            parser.add_argument(
                "--output_dir",
                type=str,
                default="results",
                help="""Path to the base directory in which output files will be
          stored. Results will automatically be sorted into subdirectories of
          the form 'testproblem/optimizer'.""")
        else:
            args["output_dir"] = output_dir

        if train_log_interval is None:
            parser.add_argument(
                "--train_log_interval",
                type=int,
                default=10,
                help="Interval of steps at which training loss is logged.")
        else:
            args["train_log_interval"] = train_log_interval

        if print_train_iter is None:
            parser.add_argument(
                "--print_train_iter",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to print mini-batch training loss to
          stdout on each (logged) iteration.""")
        else:
            args["print_train_iter"] = print_train_iter

        if no_logs is None:
            parser.add_argument(
                "--no_logs",
                action="store_const",
                const=True,
                default=False,
                help="""Add this flag to not save any json logging files.""")
        else:
            args["no_logs"] = no_logs

        # Optimizer hyperparams
        for hp in self._hyperparams:
            hp_name = hp["name"]
            if hp_name in optimizer_hyperparams:
                args[hp_name] = optimizer_hyperparams[hp_name]
            else:  # hp_name not in optimizer_hyperparams
                hp_type = hp["type"]
                if "default" in hp:
                    hp_default = hp["default"]
                    parser.add_argument(
                        "--{0:s}".format(hp_name),
                        type=hp_type,
                        default=hp_default,
                        help="""Hyperparameter {0:s} of {1:s} ({2:s};
              defaults to {3:s}).""".format(hp_name, self._optimizer_name,
                                            str(hp_type), str(hp_default)))
                else:
                    parser.add_argument(
                        "--{0:s}".format(hp_name),
                        type=hp_type,
                        required=True,
                        help="Hyperparameter {0:s} of {1:s} ({2:s}).".format(
                            hp_name, self._optimizer_name, str(hp_type)))

        # Get the command line arguments and add them to the ``args`` dict. Then
        # call the _run function with those arguments.
        cmdline_args = vars(parser.parse_args())
        args.update(cmdline_args)
        self._run(**args)

    def _run(self, testproblem, weight_decay, batch_size, num_epochs,
             learning_rate, lr_sched_epochs, lr_sched_factors, random_seed,
             data_dir, output_dir, train_log_interval, print_train_iter,
             no_logs, **optimizer_hyperparams):
        """Performs the actual run, given all the arguments.

        Args:
            testproblem (str): Name of the test problem.
            weight_decay (float): Weight decay factor.
            batch_size (int): Batch size.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Initial learning rate.
            lr_sched_epochs (list): Epochs at which to change learning rate.
            lr_sched_factors (list): Factors by which to change learning rate.
            random_seed (int): Random seed.
            data_dir (str): Data directory path.
            output_dir (str): Output directory path.
            train_log_interval (int): Logging interval.
            print_train_iter (bool): Whether to print training iterations.
            no_logs (bool): Whether to skip saving logs.
            **optimizer_hyperparams: Additional optimizer hyperparameters.
        """

        # Set data directory of DeepOBS.
        if data_dir is not None:
            config.set_data_dir(data_dir)

        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            # For reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Note: MPS uses the same manual_seed as CPU, no additional seeding needed

        # Find testproblem by name and instantiate with batch size and weight decay.
        try:
            testproblem_mod = importlib.import_module(testproblem)
            testproblem_cls = getattr(testproblem_mod, testproblem)
            print("Loading local testproblem.")
        except:
            testproblem_cls = getattr(testproblems, testproblem)

        if weight_decay is not None:
            tproblem = testproblem_cls(batch_size, weight_decay)
        else:
            tproblem = testproblem_cls(batch_size)

        # Set up the testproblem (creates dataset and model)
        tproblem.set_up()

        # Move model to device
        device = tproblem.device
        print(f"Using device: {device}")

        # Create optimizer with learning rate and hyperparameters
        # Note: In PyTorch, we handle weight decay through the optimizer parameter
        # rather than manually adding L2 loss
        opt = self._optimizer_class(
            tproblem.model.parameters(),
            lr=learning_rate,
            **optimizer_hyperparams
        )

        # Create learning rate schedule
        lr_schedule = runner_utils.make_lr_schedule(
            learning_rate, lr_sched_epochs, lr_sched_factors)

        # Create output folder
        if not no_logs:
            run_folder_name, file_name = runner_utils.make_run_name(
                weight_decay, batch_size, num_epochs, learning_rate,
                lr_sched_epochs, lr_sched_factors, random_seed,
                **optimizer_hyperparams)
            directory = os.path.join(output_dir, testproblem, self._optimizer_name,
                                     run_folder_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Lists to track train/test loss and accuracy.
        train_losses = []
        test_losses = []
        minibatch_train_losses = []
        train_accuracies = []
        test_accuracies = []

        # Helper function to evaluate on a data loader
        def evaluate(data_loader, is_test=True):
            """Computes average loss and accuracy in the evaluation phase.

            Args:
                data_loader: The DataLoader to evaluate on.
                is_test (bool): Whether this is the test set (for logging).

            Returns:
                tuple: (average_loss, average_accuracy)
                    - average_accuracy is None if the problem doesn't compute accuracy
            """
            tproblem.model.eval()
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0
            has_accuracy = False

            with torch.no_grad():
                for batch in data_loader:
                    # Get loss and accuracy for this batch
                    loss, accuracy = tproblem.get_batch_loss_and_accuracy(
                        batch, reduction='mean'
                    )

                    # Note: Evaluation metrics should only include prediction loss,
                    # not regularization. Regularization is part of the training
                    # objective, not a measure of model performance.
                    total_loss += loss.item()

                    if accuracy is not None:
                        has_accuracy = True
                        total_acc += accuracy.item()

                    num_batches += 1

            # Compute averages
            avg_loss = total_loss / max(num_batches, 1)
            avg_acc = (total_acc / max(num_batches, 1)) if has_accuracy else None

            # Log results
            msg = "TEST:" if is_test else "TRAIN:"
            if avg_acc is not None:
                print("{0:s} loss {1:g}, acc {2:f}".format(msg, avg_loss, avg_acc))
            else:
                print("{0:s} loss {1:g}".format(msg, avg_loss))

            return avg_loss, avg_acc

        # Start of training loop.
        for epoch in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            print("********************************")
            print("Evaluating after {0:d} of {1:d} epochs...".format(
                epoch, num_epochs))

            # Evaluate on training set
            train_loss, train_acc = evaluate(tproblem.dataset.train_eval_loader, is_test=False)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            # Evaluate on test set
            test_loss, test_acc = evaluate(tproblem.dataset.test_loader, is_test=True)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)

            print("********************************")

            # Break from train loop after the last round of evaluation
            if epoch == num_epochs:
                break

            # Update learning rate if scheduled
            if epoch in lr_schedule:
                new_lr = lr_schedule[epoch]
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
                print("Setting learning rate to {0:f}".format(new_lr))

            # Training
            tproblem.model.train()
            step_count = 0

            for batch in tproblem.dataset.train_loader:
                # Get loss for this batch (per-example losses, then mean)
                losses, accuracy = tproblem.get_batch_loss_and_accuracy(
                    batch, reduction='none'
                )
                loss = losses.mean()

                # Add regularization loss
                reg_loss = tproblem.get_regularization_loss()
                total_loss = loss + reg_loss

                # Backward pass and optimizer step
                opt.zero_grad()
                total_loss.backward()
                opt.step()

                # Log training loss at specified intervals
                if step_count % train_log_interval == 0:
                    minibatch_train_losses.append(total_loss.item())
                    if print_train_iter:
                        print("Epoch {0:d}, step {1:d}: loss {2:g}".format(
                            epoch, step_count, total_loss.item()))

                step_count += 1

        # --- End of training loop.

        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses
        }

        # Only add accuracies if they are computed (not None)
        # Check if the first accuracy is not None (indicating the problem computes accuracy)
        if train_accuracies and train_accuracies[0] is not None:
            output["train_accuracies"] = train_accuracies
            output["test_accuracies"] = test_accuracies

        # Put all run parameters into output dictionary.
        output["optimizer"] = self._optimizer_name
        output["testproblem"] = testproblem
        output["weight_decay"] = weight_decay
        output["batch_size"] = batch_size
        output["num_epochs"] = num_epochs
        output["learning_rate"] = learning_rate
        output["lr_sched_epochs"] = lr_sched_epochs
        output["lr_sched_factors"] = lr_sched_factors
        output["random_seed"] = random_seed
        output["train_log_interval"] = train_log_interval

        # Add optimizer hyperparameters as a sub-dictionary.
        output["hyperparams"] = optimizer_hyperparams

        # Dump output into json file.
        if not no_logs:
            output_path = os.path.join(directory, file_name + ".json")
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to: {output_path}")
