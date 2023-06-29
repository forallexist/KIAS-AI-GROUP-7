import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import math 
import os
import wandb

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, CosineAnnealingLR

from data.generator2d import generate_grid2d, true_green
from utils.helpers import normalized_bump

from src.models import DRM 
from trainer import train_basic

def main():


    device = torch.device('cuda:0')
    
    exp_index = 2
    lr = 1e-3

    file_dir = os.getcwd()
    # save_model_path = os.path.join(file_dir, "models/")
    save_figure_path = os.path.join(file_dir, "figure/")
    
    save_fig = False   

    start = 0
    end = 1
    resolution = 50

    epochs = 200000
    beta = 50

    wandb.init(
        # set the wandb project where this run will be logged
        project="kias_summer_school_23_team7",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "DRM",
        "dataset": "Dirac delta",
        "epochs": epochs,
        "beta" : beta, 
        "resolution": resolution
        }
    )

    _, _, xy_grid, xy_bdry, _ = generate_grid2d(
        resolution=resolution, 
        x_start=start, x_end=end, 
        y_start=start, y_end=end,
        device=device)

    xy_grid.requires_grad_(True)

    # for green function
    # parameters in the PDE
    k = 1
    sigma = 1/math.sqrt(2)

    mollifier = lambda x: normalized_bump(x, sigma)

    x_center = torch.tensor(0.3)
    y_center = torch.tensor(0.7)
    N_cutoff = resolution
    xy_grid_true = xy_grid.clone().cpu().detach()
    xx_true = xy_grid_true[:,0]
    yy_true = xy_grid_true[:,1]

    zz_true = true_green(xx_true, yy_true, x_center, y_center, N_cutoff, k=k)

    # Prepare for training
    


    network = DRM(hidden_dims=[2,16,16,16,1]).to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    train_basic(
        model=network,
        grid_data=xy_grid,
        boundary_data=xy_bdry,
        target=zz_true,
        mollifier=mollifier,
        optimizer=optimizer,
        scheduler = scheduler,
        k=k,
        epoch=epochs,
        beta = beta,
        device=device,
        exp_tag=exp_index,
        wandb_logging = True
        )
    
    wandb.finish()

    if save_fig:
        pred = network(xy_grid).detach().cpu()

        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121, projection = '3d')
        ax2 = fig.add_subplot(122,projection = '3d')
        
        def rotate_fig(angle):
            ax1.view_init(azim=angle)
            ax2.view_init(azim=angle)

        surf = ax1.scatter(xx_true, yy_true, zz_true.view(-1,1), c= zz_true, s = 1, label="True Green function of dirac delta")
        surf2 = ax2.scatter(xx_true, yy_true, pred.view(-1,1), c= pred, s = 1, label = "Prediction")
        
        rot_animation = animation.FuncAnimation(fig, rotate_fig, frames=np.arange(0,362,2),interval=100)
        
        fig.colorbar(surf)
        fig.colorbar(surf2)

        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax1.set_zlabel('z axis')
        
        ax2.set_xlabel('x axis')
        ax2.set_ylabel('y axis')
        ax2.set_zlabel('z axis')
        
        plt.legend()
        plt.tight_layout()
        rot_animation.save(save_figure_path + f"ouput_ani_{exp_index}.gif")
        plt.show()



    

# %%
if __name__ == '__main__':
    
    main()
    
    
# %%
