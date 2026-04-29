import torch


def predict_single(model, data, device):
    """Executes a single forward pass without batching overhead."""
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(data.x.float(),
                     data.edge_index,
                     data.edge_attr.float(),
                     data.node_index,
                     data.mol_formal_charge,
                     data.center_formal_charge,
                     data.batch
                     )

    return pred.item()
