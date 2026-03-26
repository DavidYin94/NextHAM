import torch
from e3nn.o3 import wigner_3j

def get_nodetarget(output_node, train_target, data):
        mse = torch.nn.MSELoss()
        energy = force = zstar = None
        energy_loss = force_loss = zstar_loss = torch.zeros(1, device = output_node.device)
        if 'E' in train_target:
            assert data.energy is not None
            energy_fea = output_node[..., 0]
            energy = energy_fea.reshape(1, -1).sum(dim=-1)
            energy_loss = mse(energy, data.energy)

        if 'F' in train_target:
            assert data.force is not None
            mask = data["edge_attr"][:,0] != 0
            force_edge0 = torch.autograd.grad(torch.sum(output_node[..., 0]), data["edge_attr"], create_graph=True, only_inputs=True, allow_unused=True)[0][mask]
            data_mask = data["edge_attr"][mask].detach()
            force_edge = force_edge0[:,[1,2,3]] + force_edge0[:,[0]] * data_mask[:,[1,2,3]] / data_mask[:,[0]]
            with torch.no_grad():
                edge_index = data["edge_index"]
            force = torch.zeros_like(data.force)
            filtered_edge_index_0 = edge_index[0][mask]
            filtered_edge_index_1 = edge_index[1][mask]
            filtered_force_edge = force_edge
            force.index_add_(0, filtered_edge_index_1, -filtered_force_edge)
            force.index_add_(0, filtered_edge_index_0, filtered_force_edge)
            force_loss = mse(force, data.force)

        if 'Z' in train_target:
            zstar_fea = output_node[..., 1:]
            assert data.zstar is not None
            wm = []
            for i in range(3):
                wm.append(wigner_3j(1, 1, i, dtype=output_node.dtype, device=output_node.device))
            wm = torch.cat(wm, dim = -1)
            zstar = torch.sum(wm[None, :, :, :] * zstar_fea[:, None, None, :], dim=-1)
            zstar = zstar.reshape(-1,9)
            zstar_loss = mse(zstar, data.zstar)
        return energy, force, zstar, energy_loss, force_loss, zstar_loss