import torch



def generate_grid2d(
        resolution:int, 
        x_start:int, x_end:int, 
        y_start:int, y_end:int, 
        device):
    
    x = torch.linspace(x_start, x_end, resolution).to(device)
    y = torch.linspace(y_start, y_end, resolution).to(device)
    xy_plane = torch.meshgrid(x,y)
    
    xy_grid = torch.cat([xy_plane[0].reshape(-1,1), xy_plane[1].reshape(-1,1)],dim=1)

    x_bdry = xy_grid[torch.logical_or((xy_grid[:,0]==1), (xy_grid[:,0]==0))]
    y_bdry = xy_grid[torch.logical_or((xy_grid[:,1]==1), (xy_grid[:,1]==0))]
    xy_bdry = torch.cat([x_bdry, y_bdry])
    u_bdry = 0*(xy_bdry[:,0] * xy_bdry[:,1]).view(-1,1)    
    
    return x, y, xy_grid, xy_bdry, u_bdry



def eigenvalue(nx, ny, k):
    return -(nx**2 + ny**2)*torch.pi**2 + k*2

def eigenfunction(x, y, nx, ny):
    return 2*torch.sin(nx*torch.pi*x.clone().detach())*torch.sin(ny*torch.pi*y.clone().detach())

def true_green(x, y, x_center, y_center, N_cutoff, k):
    value = 0
    for n_x in range(1, N_cutoff):
        for n_y in range(1, N_cutoff):
            value += eigenfunction(x, y, n_x, n_y) \
                     * eigenfunction(x_center, y_center, n_x, n_y)\
                     /eigenvalue(n_x, n_y, k=k)

    return value.clone().detach()

