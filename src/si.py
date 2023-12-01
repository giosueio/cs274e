import numpy as np
import torch
from torchdiffeq import odeint
from util.util import plot_loss_curve, plot_samples

import time

class SI:
    '''
    Stochastic Interpolant base class.
    '''

    def __init__(self, model, device='cpu', dtype=torch.double):
        self.model = model
        self.device = device
        self.dtype = dtype

    def simulate(self, t, x_0, x_1):
        '''
        Simulate x_t from the stochastic interpolant, s.t. x_t = I(t, x_0, x_1) + gamma(t)*z, z ~ N(0, I)
        '''
        z = torch.randn_like(x_0)
        x_t = self.I(t, x_0, x_1) + self.gamma(t)*z
        return x_t, z
            
    def I(self, t, x_0, x_1):
        '''
        Deterministic component of the stochastic interpolant, i.e. I(t, x_0, x_1)
        '''
        AssertionError('Not implemented')

    def gamma(self, t):
        '''
        Random component of the stochastic interpolant, i.e. gamma(t)*z, z ~ N(0, I)
        '''
        AssertionError('Not implemented')
    
    def v(self, t, x_0, x_1):
        '''
        Conditional vector field, i.e. v(t,x) = E[d_t I(t, x_0, x_1) | x_t = x]
        '''
        AssertionError('Not implemented')

    def dgamma(self, t):
        '''
        Derivative of gamma(t)
        '''
        AssertionError('Not implemented')

    def b(self, t, x_0, x_1, z):
        '''
        Velocity term, i.e. b(t, x_t) = v(t, x_0, x_1) + dgamma(t)*gamma(t)*s(t)
        '''
        return self.v(t, x_0, x_1) + self.dgamma(t)*z

    def s(self, t, z):
        '''
        Score term, i.e. s(t,x_t) = - E[z|x_t = x] / gamma(t)
        '''
        # TODO: find best fix when gamma(t) = 0, see section 6.1 of Albergo et al. 2023
        return -z/(self.gamma(t) + 1e-8)
    
    def compute_loss(self, loss_type, model_out, t, x_0, x_1, z):
        if loss_type == 'score':
            target = self.s(t, z)
        elif loss_type == 'velocity':
            target = self.b(t, x_0, x_1, z)
        elif loss_type == 'cvf':
            target = self.v(t, x_0, x_1)
        elif loss_type == 'noise':
            target = z
        return ((model_out - target)**2).mean()

    def train(self, train_loader, optimizer, epochs, loss_type='cvf',
                scheduler=None, test_loader=None, eval_int=0, 
                save_int=0, generate=False, save_path=None):

        tr_losses = []
        te_losses = []
        eval_eps = []
        evaluate = (eval_int > 0) and (test_loader is not None)

        self.loss_type = loss_type
        self.one_sided = False

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
                if first or not self.one_sided:
                    try:
                        x_0, x_1 = batch
                        x_0, x_1 = x_0.to(device), x_1.to(device)
                    except ValueError:
                        # When train_loader contains only one dataset, resort to one-sided interpolants (i.e. rho_1 is Gaussian)
                        x_0 = batch.to(device)
                        x_1 = torch.randn_like(x_0).to(device)
                        self.one_sided = True
                else:
                    x_0 = batch.to(device)
                    x_1 = torch.randn_like(x_0).to(device)
                
                batch_size = x_0.shape[0]

                if first:
                    self.n_channels = x_0.shape[1]
                    self.train_dims = x_0.shape[2:]
                    first = False

                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                # Simulate x_t
                x_t, z = self.simulate(t, x_0, x_1)
                # Get model output
                model_out = model(x_t, t)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = self.compute_loss(loss_type, model_out, t, x_0, x_1, z)
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
                            if not self.one_sided:
                                x_0, x_1 = batch
                                x_0, x_1 = x_0.to(device), x_1.to(device)
                            else:
                                x_0 = batch.to(device)
                                x_1 = torch.randn_like(x_0).to(device)

                            batch_size = x_0.shape[0]

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)
                            # Simulate x_t
                            x_t = self.simulate(t, x_0, x_1)
                            # Get model output
                            model_out = model(x_t, t)

                            # Evaluate loss
                            loss = self.compute_loss(loss_type, model_out, t, x_0, x_1, z)
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

    def sample_r(self, n_samples):
        '''
        Sample from the stochastic interpolant, starting from x_1.
        '''
        AssertionError('Not implemented')

    def v_to_b_model(self, model):
        '''
        Given a model of the conditional vector field v(t,x), return a model of the velocity field b(t,x_t).
        '''
        AssertionError('Not implemented')

    def s_to_b_model(self, model):
        '''
        Given a model of the score term s(t,x_t), return a model of the velocity field b(t,x_t).
        '''
        AssertionError('Not implemented')

    def eta_to_b_model(self, model):
        '''
        Given a model of the noise term z, return a model of the velocity field b(t,x_t).
        '''
        AssertionError('Not implemented')

    def get_b_model(self, direction):
        '''
        Return the model of the velocity field b(t, x_t), depending on the direction of the ODE and on the loss type.
        '''
        if self.loss_type == 'velocity':
            model = self.model
        elif self.loss_type == 'cvf':
            model = self.v_to_b_model(self.model)
        elif self.loss_type == 'score':
            model = self.s_to_b_model(self.model)
        elif self.loss_type == 'noise':
            model = self.eta_to_b_model(self.model)

        if direction == 'f':
            return model
        elif direction == 'r':
            return lambda x, t: -model(x, 1-t) # TODO: not sure if this is correct. might be -model(x, t)
        
        
    @torch.no_grad()
    def sample(self, x_initial, direction='f', n_samples=1, n_eval=2, return_path=False, rtol=1e-5, atol=1e-5):
        '''
        Sample from the stochastic interpolant using an ODE, starting from x_initial.

        Arguments:
        - x_initial: [batch_size, n_channels, *dims]
        - direction: 'f' or 'r', forward or reverse in time
        - n_samples: number of samples to generate
        - n_eval: number of timesteps to evaluate
        - return_path: if True, return the entire path of samples, otherwise just the final sample
        - rtol, atol: tolerances for odeint
        '''        

        t = torch.linspace(0, 1, n_eval, device=self.device)
        inital_batch_size = x_initial.shape[0]

        if n_samples > inital_batch_size:
            # if x_initial has less samples than n_samples, repeat x_initial at random to get n_samples
            n_new_samples = n_samples - inital_batch_size
            x_initial = torch.cat([x_initial, x_initial[torch.randperm(inital_batch_size)[:n_new_samples]]], dim=0)

        b_model = self.get_b_model(self, direction)
        method = 'dopri5'
        out = odeint(b_model, x_initial, t, method=method, rtol=rtol, atol=atol)

        if return_path:
            return out
        else:
            return out[-1]

class SLI(SI):
    '''
    Spatially Linear Interpolant base class, i.e. x(t) = a(t)*x_0 + b(t)*x_1
    '''

    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)

    def a(self, t):
        AssertionError('Not implemented')
    def b(self, t):
        AssertionError('Not implemented')
    def da(self, t):
        AssertionError('Not implemented')
    def db(self, t):
        AssertionError('Not implemented')

    def I(self, t, x_0, x_1):
        return self.a(t)*x_0 + self.b(t)*x_1
    def dI(self, t, x_0, x_1):
        return self.da(t)*x_0 + self.db(t)*x_1
    def gamma(self, t):
        return torch.zeros_like(t)
    def dgamma(self, t):
        return torch.zeros_like(t)
    
    def eta_to_b_model(self, model):
        '''
        Given a model of the noise term z, return a model of the velocity field b(t,x_t). Use only for one-sided interpolants.
        See 6.2 of Albergo et al. 2023, "A denoiser is all you need for spatially-linear one-sided interpolants"
        '''
        model = lambda x, t: self.da(t)*model(x, t) + self.db(t)/self.b(t)*(x - self.a(t)* self.model(x,t))
        return model

class LinearInterpolant(SLI):
    '''
    Linear Interpolant, i.e. x(t) = (1-t)*x_0 + t*x_1
    References: Liu et al. 2023, Flow straight and fast: Learning to generate and transfer data with rectified flow
                Lipman et al. 2023, Flow matching for generative modeling
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)
    def a(self, t):
        return 1 - t
    def b(self, t):
        return t
    def da(self, t):
        return -torch.ones_like(t)
    def db(self, t):
        return torch.ones_like(t)
    
class TrigLinearInterpolant(SLI):
    '''
    Trigonometric Linear Interpolant, i.e. x(t) = sin(pi*t/2)*x_0 + cos(pi*t/2)*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)
    def a(self, t):
        return torch.sin(np.pi*t/2)
    def b(self, t):
        return torch.cos(np.pi*t/2)
    def da(self, t):
        return np.pi/2 * torch.cos(np.pi*t/2)
    def db(self, t):
        return -np.pi/2 * torch.sin(np.pi*t/2)
   
class PolynomialInterpolant(SLI):
    '''
    Polynomial Interpolant of order p, i.e. x(t) = (1-t)^p*x_0 + t^p*x_1
    '''
    def __init__(self, p, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
        self.p = p

    def a(self, t):
        return (1-t)**self.p
    def b(self, t):
        return t**self.p
    def da(self, t):
        return -self.p*(1-t)**(self.p-1)
    def db(self, t):
        return self.p*t**(self.p-1)
    
class DiffusionInterpolant(SLI):
    '''
    Variance-preserving diffusion interpolant, i.e. x(t) = sqrt(1-t^2)*x_0 + t*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
    def a(self, t):
        return np.sqrt(1-t**2)
    def b(self, t):
        return t
    def da(self, t):
        return -t/np.sqrt(1-t**2)
    def db(self, t):
        return np.ones_like(t)

class EncoderDecoderInterpolant(SLI):
    '''
    Encoder-decoder interpolant, i.e. x(t) = cos^2(pi*t)*1_{[0,.5)}(t)*x_0 + cos^2(pi*t)*1_{[.5,1]}(t)*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
    def a(self, t):
        a = torch.zeros_like(t)
        a[t < .5] = torch.cos(np.pi*t[t < .5])**2
        return a
    def b(self, t):
        b = torch.zeros_like(t)
        b[t >= .5] = torch.cos(np.pi*t[t >= .5])**2
        return b
    def da(self, t):
        da = torch.zeros_like(t)
        da[t < .5] = -2*np.pi*torch.cos(np.pi*t[t < .5])*torch.sin(np.pi*t[t < .5])
        return da
    def db(self, t):
        db = torch.zeros_like(t)
        db[t >= .5] = -2*np.pi*torch.cos(np.pi*t[t >= .5])*torch.sin(np.pi*t[t >= .5])
        return db
        

class MirrorInterpolant(SLI):
    '''
    Mirror interpolant, i.e. x(t) = x_0 (recommended to add noise :-) )
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
        make_noisy(self)

    def a(self, t):
        return torch.ones_like(t)
    def b(self, t):
        return torch.zeros_like(t)
    def da(self, t):
        return torch.zeros_like(t)
    def db(self, t):
        return torch.zeros_like(t)

def make_noisy(SI):
    '''
    Add the gamma(t)=np.sqrt(2*t(1-t)) component to an interpolant
    '''
    SI.gamma = lambda t: np.sqrt(2*t(1-t))
    SI.dgamma = lambda t: (1 - 2*t)/np.sqrt(2*t(1-t))
    return SI

def make_sinsq_noisy(SI):
    '''
    Add the gamma(t)=np.sin(np.pi*t)**2 component to an interpolant
    '''
    SI.gamma = lambda t: np.sin(np.pi*t)**2
    SI.dgamma = lambda t: 2*np.pi*np.sin(np.pi*t)*np.cos(np.pi*t)
    return SI
    