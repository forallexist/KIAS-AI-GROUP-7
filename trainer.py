import torch
import torch.nn as nn
from utils.helpers import derivative
import wandb


def train_basic(
        model, 
        grid_data, boundary_data, 
        target, 
        mollifier,
        optimizer,
        scheduler, 
        beta,
        k, 
        epoch, 
        device,
        exp_tag,
        wandb_logging
        ):
    

    loss_list = []
    error_list = []

    model.train()
    for i in range(1, epoch+1) :
        optimizer.zero_grad()
        output = model(grid_data)
        output_bdry = model(boundary_data)
        grad_output = derivative(output, grid_data, device=device)
        
        integrand_interior = 0.5*torch.pow(torch.norm(grad_output, dim=1), 2).view(-1,1) \
                             + 0.5*torch.pow(k*output, 2).view(-1,1) \
                             - output * mollifier(torch.sqrt(grid_data[:,0]**2 + grid_data[:,1]**2))
            
        integrand_bdry = (output_bdry-boundary_data)**2
        
        loss_ge = integrand_interior.mean() 
        loss_bdry = integrand_bdry.mean() 
        
        loss = loss_ge + beta * loss_bdry
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = loss.item() 
        loss_list.append(loss_value)
        if wandb_logging:
            loss_f = nn.MSELoss()
            pred = model(grid_data).detach().cpu()
            l2loss = loss_f(target.view(-1), pred.view(-1)).item()
            
            wandb.log({
                "total_loss": loss, 
                "interior_loss": loss_ge.item(), 
                "boundary_loss":loss_bdry.item(),
                "MSE": l2loss
                })
            
        if not i % 100 :
            print(
                '''EPOCH : %6d/%6d | Loss_ge : %8.7f | Loss_bdry : %8.7f  |  lr : %8.7f''' 
                %(i, epoch, loss_ge.item(), loss_bdry.item(), scheduler.get_last_lr()[-1]))
            #clear_output(wait=True)
    print('Training Finished.')
    torch.save(obj=model, f = f'DRM_2d_{exp_tag}.pt')
    
    with torch.no_grad():
        pred = model(grid_data).detach().cpu()
        error_l2 = nn.MSELoss()(target.view(-1), pred.view(-1))
        error_list.append(error_l2.item())
        print(f"L2 error is {error_l2.item()}")
