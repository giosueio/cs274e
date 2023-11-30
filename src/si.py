import numpy as np
import torch
from torchdiffeq import odeint
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples

import time

class StochasticInterpolant:
    def __init__(self, model, device='cpu', dtype=torch.double):
        self.model = model
        self.device = device
        self.dtype = dtype

    def simulate(self, t, x_batch):
        '''
        Simulate x_t from the stochastic interpolant, s.t. x_t = I(t, x_0, x_1) + gamma(t)*z, z ~ N(0, I)
        '''
        AssertionError('Not implemented')
            
    def I(self, t, x_batch):
        '''
        Deterministic component of the stochastic interpolant, i.e. I(t, x_0, x_1)
        '''
        AssertionError('Not implemented')

    def gamma(self, t):
        '''
        Random component of the stochastic interpolant, i.e. gamma(t)*z, z ~ N(0, I)
        '''
        AssertionError('Not implemented')
    
    def b(self, t, x_batch, x_t):
        '''
        Conditional vector field, i.e. b(t, x_0, x_t)
        '''
        AssertionError('Not implemented')

    def train(self, train_loader, optimizer, epochs, 
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        model = self.model
        device = self.device
        dtype = self.dtype

        first = True
        for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for batch in train_loader:
                batch = (x.to(device) for x in batch)
                batch_size = batch[0].shape[0]

                if first:
                    self.n_channels = batch[0].shape[1]
                    self.train_dims = batch[0].shape[2:]
                    first = False

                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                # Simulate x_t
                x_t = self.simulate(t, batch)
                # Get conditional vector fields
                target = self.b(t, batch, x_t)

                x_t = x_t.to(device)
                target = target.to(device)         

                # Get model output
                model_out = model(x_t, t)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = ((model_out - target)**2).mean() 
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()


            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            ##### EVAL LOOP
            if eval_int > 0 and (ep % eval_int == 0):
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()

                    if evaluate:
                        te_loss = 0.0
                        for batch in test_loader:
                            batch = (x.to(device) for x in batch)
                            batch_size = batch[0].shape[0]

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)
                            # Simulate p_t(x | x_1)
                            x_t = self.simulate(t, batch)
                            # Get conditional vector fields
                            target = self.b(t, batch, x_t)
                        
                            x_t = x_t.to(device)
                            target = target.to(device)         
                            model_out = model(x_t, t)

                            loss = torch.mean( (model_out - target)**2 )

                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')


                    # genereate samples during training?
                    if generate:
                        samples = self.sample(self.train_dims, n_channels=self.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')


            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(model.state_dict(), save_path / f'epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')

    def sample_f(self, n_samples):
        '''
        Sample from the stochastic interpolant, starting from x_0.
        '''
        AssertionError('Not implemented')

    def sample_b(self, n_samples):
        '''
        Sample from the stochastic interpolant, starting from x_1.
        '''
        AssertionError('Not implemented')

    @torch.no_grad()
    def sample(self, x_initial, direction='f', n_samples=1, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5):
        '''
        Sample from the stochastic interpolant, starting from x_initial.

        Arguments:
        - x_initial: [batch_size, n_channels, *dims]
        - direction: 'f' or 'b', forward or backward in time
        - n_samples: number of samples to generate
        - n_eval: number of timesteps to evaluate
        - return_path: if True, return the entire path of samples, otherwise just the final sample
        - rtol, atol: tolerances for odeint
        '''        

        t = torch.linspace(0, 1, n_eval, device=self.device)
        inital_batch_size = x_initial.shape[0]

        if n_samples > inital_batch_size:
            # if x_initial has less samples than n_samples, repeat x_initial at random along batch dimension to get n_samples
            x_initial = x_initial.repeat((n_samples // inital_batch_size) + 1, 1, 1, 1)[:n_samples]

        method = 'dopri5'
        out = odeint(self.model, x_initial, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]

class GaussianSI(StochasticInterpolant):
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)
        self.rho1 = torch.distributions.Normal(0, 1)

        # TODO
