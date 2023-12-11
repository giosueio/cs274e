import numpy as np
import torch
from tqdm import tqdm
from torchdiffeq import odeint
from src.util.util import optimal_WD_matching, plot_loss_curve, plot_samples

import time

class SI:
    '''
    Stochastic Interpolant base class.
    '''

    def __init__(self, model, device='cpu', dtype=torch.float):
        self.model = model
        self.device = device
        self.dtype = dtype

    def simulate(self, t, x_0, x_1):
        '''
        Simulate x_t from the stochastic interpolant, s.t. x_t = I(t, x_0, x_1) + gamma(t)*z, z ~ N(0, I)
        '''
        z = torch.randn_like(x_0)
        batch_size = x_0.shape[0]

        x_t = self.I(t, x_0, x_1) + self.gamma(t).view(batch_size,1,1,1)*z
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
        Velocity term, i.e. b(t, x_t) = v(t, x_0, x_1) + dgamma(t)*z
        '''
        batch_size = x_0.shape[0]
        return self.v(t, x_0, x_1) + self.dgamma(t).view(batch_size,1,1,1)*z

    def s(self, t, z):
        '''
        Score term, i.e. s(t,x_t) = - E[z|x_t = x] / gamma(t)
        '''
        # TODO: find best fix when gamma(t) = 0, see section 6.1 of Albergo et al. 2023
        return -z/(self.gamma(t) + 1e-8)
    
    def straightness(self, b, x_0, x_1):
        '''
        Straightness of the interpolant, i.e. ||b(t, x_t)||_2^2
        '''
        S = b.norm()**2 + (x_1 - x_0).norm()**2
        return S

    def compute_loss(self, loss_type, model_out, t, x_0, x_1, z, lmbd=0.):
        if loss_type == 'score':
            target = self.s(t, z)
        elif loss_type == 'velocity':
            target = self.b(t, x_0, x_1, z)
            if lmbd>0:
                S = self.straightness(model_out, x_0, x_1)
                return ((model_out - target)**2).mean() + lmbd*S
        elif loss_type == 'cvf':
            target = self.v(t, x_0, x_1)
        elif loss_type == 'noise':
            target = z
        return ((model_out - target)**2).mean()

    def train(self, train_loader, optimizer, epochs, loss_type='cvf',
                scheduler=None, test_loader=None, eval_int=0, mcls=False,
                encoder=None, save_int=0, save_path=None, lmbd=0.):
        
        if mcls:
            assert encoder is not None, 'Encoder must be provided for MCLS training'
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
        for ep in tqdm(range(1, epochs+1)):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0

            for batch in train_loader:
                if first or not self.one_sided:
                    try:
                        x_0, x_1 = batch
                        x_0, x_1 = x_0.to(device), x_1.to(device)
                        if mcls:
                            x_1 = optimal_WD_matching(x_0, x_1, encoder)
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
                model_out = model(t, x_t)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = self.compute_loss(loss_type, model_out, t, x_0, x_1, z, lmbd)
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
                            model_out = model(t, x_t)

                            # Evaluate loss
                            loss = self.compute_loss(loss_type, model_out, t, x_0, x_1, z)
                            te_loss += loss.item()
                            
                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')

            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(model.state_dict(), f'{save_path}/epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, f'{save_path}/loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, f'{save_path}/loss.pdf')

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

    def get_b_model(self):
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

        return model
       
        
    @torch.no_grad()
    def sample(self, x_initial, direction='f', n_samples=None, n_eval=2, device=None,
               return_path=False, rtol=1e-5, atol=1e-5, lower_tlim=0, upper_tlim=1):
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
        if device is None:
            device = self.device
        if n_samples is None:
            n_samples = x_initial.shape[0]
        if direction == 'f':
            t = torch.linspace(lower_tlim, upper_tlim, n_eval, device=device)
        elif direction == 'r':
            t = torch.linspace(upper_tlim, lower_tlim, n_eval, device=device)
        inital_batch_size = x_initial.shape[0]

        if n_samples > inital_batch_size:
            # if x_initial has less samples than n_samples, repeat x_initial at random to get n_samples
            n_new_samples = n_samples - inital_batch_size
            x_initial = torch.cat([x_initial, x_initial[torch.randperm(inital_batch_size)[:n_new_samples]]], dim=0)

        x_initial = x_initial.to(device)
        b_model = self.get_b_model().to(device)
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

    def alpha(self, t):
        AssertionError('Not implemented')
    def beta(self, t):
        AssertionError('Not implemented')
    def dalpha(self, t):
        AssertionError('Not implemented')
    def dbeta(self, t):
        AssertionError('Not implemented')

    def I(self, t, x_0, x_1):
        batch_size = x_0.shape[0]
        return self.alpha(t).view(batch_size,1,1,1)*x_0 + self.beta(t).view(batch_size,1,1,1)*x_1
    def v(self, t, x_0, x_1):
        batch_size = x_0.shape[0]
        return self.dalpha(t).view(batch_size,1,1,1)*x_0 + self.dbeta(t).view(batch_size,1,1,1)*x_1
    def gamma(self, t):
        return torch.zeros_like(t)
    def dgamma(self, t):
        return torch.zeros_like(t)
    
    # def eta_to_b_model(self, model):
    #     '''
    #     Given a model of the noise term z, return a model of the velocity field b(t,x_t). Use only for one-sided interpolants.
    #     See 6.2 of Albergo et al. 2023, "A denoiser is all you need for spatially-linear one-sided interpolants"
    #     '''
    #     b_model = torch.nn.Module()
    #     b_model.forward = lambda t, x: self.da(t)*model(t,x) + self.db(t)/self.b(t)*(x - self.a(t)* self.model(t,x))
    #     return b_model

class LinearInterpolant(SLI):
    '''
    Linear Interpolant, i.e. x(t) = (1-t)*x_0 + t*x_1
    References: Liu et al. 2023, Flow straight and fast: Learning to generate and transfer data with rectified flow
                Lipman et al. 2023, Flow matching for generative modeling
    N.B.: If we add noise by make_noisy(self) and define optimal couplings, then this interpolant is the Schrödinger Bridge interpolant from Tong et al. 2023
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)
    def alpha(self, t):
        return 1 - t
    def beta(self, t):
        return t
    def dalpha(self, t):
        return -torch.ones_like(t)
    def dbeta(self, t):
        return torch.ones_like(t)
    
class TrigLinearInterpolant(SLI):
    '''
    Trigonometric Linear Interpolant, i.e. x(t) = sin(pi*t/2)*x_0 + cos(pi*t/2)*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device=device, dtype=dtype)
    def alpha(self, t):
        return torch.sin(np.pi*t/2)
    def beta(self, t):
        return torch.cos(np.pi*t/2)
    def dalpha(self, t):
        return np.pi/2 * torch.cos(np.pi*t/2)
    def dbeta(self, t):
        return -np.pi/2 * torch.sin(np.pi*t/2)
   
class PolynomialInterpolant(SLI):
    '''
    Polynomial Interpolant of order p, i.e. x(t) = (1-t)^p*x_0 + t^p*x_1
    '''
    def __init__(self, model, p=2, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
        self.p = p

    def alpha(self, t):
        return (1-t)**self.p
    def beta(self, t):
        return t**self.p
    def dalpha(self, t):
        return -self.p*(1-t)**(self.p-1)
    def dbeta(self, t):
        return self.p*t**(self.p-1)
    
class DiffusionInterpolant(SLI):
    '''
    Variance-preserving diffusion interpolant, i.e. x(t) = sqrt(1-t^2)*x_0 + t*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
    def alpha(self, t):
        return torch.sqrt(1-t**2)
    def beta(self, t):
        return t
    def dalpha(self, t):
        return -t/torch.sqrt(1-t**2)
    def dbeta(self, t):
        return torch.ones_like(t)

class EncoderDecoderInterpolant(SLI):
    '''
    Encoder-decoder interpolant, i.e. x(t) = cos^2(pi*t)*1_{[0,.5)}(t)*x_0 + cos^2(pi*t)*1_{[.5,1]}(t)*x_1
    '''
    def __init__(self, model, device='cpu', dtype=torch.double):
        super().__init__(model, device, dtype)
    def alpha(self, t):
        a = torch.zeros_like(t)
        a[t < .5] = torch.cos(np.pi*t[t < .5])**2
        return a
    def beta(self, t):
        b = torch.zeros_like(t)
        b[t >= .5] = torch.cos(np.pi*t[t >= .5])**2
        return b
    def dalpha(self, t):
        da = torch.zeros_like(t)
        da[t < .5] = -2*np.pi*torch.cos(np.pi*t[t < .5])*torch.sin(np.pi*t[t < .5])
        return da
    def dbeta(self, t):
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

    def alpha(self, t):
        return torch.ones_like(t)
    def beta(self, t):
        return torch.zeros_like(t)
    def dalpha(self, t):
        return torch.zeros_like(t)
    def dbeta(self, t):
        return torch.zeros_like(t)

def make_noisy(SI, noise_coeff):
    '''
    Add the gamma(t)=torch.sqrt(2*noise_coeff*t*(1-t)) component to an interpolant
    Add the gamma(t)=torch.sqrt(2*noise_coeff*t*(1-t)) component to an interpolant
    '''
    SI.noise_coeff = noise_coeff
    SI.gamma = lambda t: torch.sqrt(2*SI.noise_coeff*t*(1-t))
    SI.dgamma = lambda t: (SI.noise_coeff*(1 - 2*t))/torch.sqrt(SI.noise_coeff*2*t*(1-t))
    return SI

def make_sinsq_noisy(SI):
    '''
    Add the gamma(t)=torch.sin(np.pi*t)**2 component to an interpolant
    '''
    SI.gamma = lambda t: torch.sin(np.pi*t)**2
    SI.dgamma = lambda t: 2*np.pi*torch.sin(np.pi*t)*torch.cos(np.pi*t)
    return SI
    

##### REFERENCES #####
# Albergo et al. 2023, Stochastic Interpolants: a unifying framework for flows and diffusions
# Liu et al. 2023, Flow straight and fast: Learning to generate and transfer data with rectified flow
# Lipman et al. 2023, Flow matching for generative modeling
# Tong et al. 2023, Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport