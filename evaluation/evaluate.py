"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/eval

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

"""

import torch
from sklearn import manifold

import utils
import visual.plt


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
    precision = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision


def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    precision = {}
    precision["all_tasks"] = [[] for _ in range(n_tasks)]
    precision["average"] = []
    precision["x_iteration"] = []
    precision["x_task"] = []
    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None,
              precision_dict=None, test_size=None, visdom=None, verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    n_tasks = len(datasets)
    precs = []
    for i in range(n_tasks):
        if i + 1 <= current_task:
            allowed_classes = list(range(classes_per_task * current_task))
            precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1))
        else:
            precs.append(0)
    average_precs = sum(
        [precs[task_id] if task_id == 0 else precs[task_id] for task_id in range(current_task)]
    ) / (current_task)

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.3f}'.format(average_precs))

    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(
            scalars=precs, names=names, iteration=iteration,
            title="Accuracy per task ({})".format(visdom["graph"]), env=visdom["env"], ylabel="precision"
        )
        if n_tasks > 1:
            visual.visdom.visualize_scalars(
                scalars=[average_precs], names=["ave precision"], iteration=iteration,
                title="Average accuracy ({})".format(visdom["graph"]), env=visdom["env"], ylabel="precision"
            )

    # Append results to [progress]-dictionary and return
    if precision_dict is not None:
        for task_id, _ in enumerate(names):
            precision_dict["all_tasks"][task_id].append(precs[task_id])
        precision_dict["average"].append(average_precs)
        precision_dict["x_iteration"].append(iteration)
        precision_dict["x_task"].append(current_task)
    return precision_dict


####--------------------------------------------------------------------------------------------------------------####

####------------------------------------------####
####----VISUALIZE EXTRACTED REPRESENTATION----####
####------------------------------------------####

def visualize_latent_space(model, X, y=None, visdom=None, verbose=False, file_path=None):
    '''Show T-sne projection of feature representation used to classify from (with each class in different color).'''

    # Set model to eval()-mode
    model.eval()

    # Compute the representation used for classification
    if verbose:
        print("Computing feature space...")
    with torch.no_grad():
        z_mean = model.feature_extractor(X)

    # Compute t-SNE embedding of latent space (unless z has 2 dimensions!)
    if z_mean.size()[1] == 2:
        z_tsne = z_mean.cpu().numpy()
    else:
        if verbose:
            print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        z_tsne = tsne.fit_transform(z_mean.cpu())
    # Plot images according to t-sne embedding
    visual.plt.plot_scatter(z_tsne[:, 0], z_tsne[:, 1], colors=y.cpu().numpy(), file_path=file_path)
    if visdom is not None:
        message = ("Visualization of extracted representation")
        visual.visdom.scatter_plot(z_tsne, title="{} ({})".format(message, visdom["graph"]),
                                   colors=y + 1 if y is not None else y, env=visdom["env"])
