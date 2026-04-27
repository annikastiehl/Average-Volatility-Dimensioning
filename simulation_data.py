#SimData
import numpy as np
from scipy.integrate import odeint
from time import time
from space_time_data import space_time_data
from scipy.stats import norm
import scipy.io 

"""class simulationData for creating data

#     DimSignal                   DimNoise
#              q(t) = sum ((x_i(t)+r_i(t))*w_i) + sum(xi_j(t)*psi_j) + n(t)
#                     i=1                         j=1
#                                  r_i                xi_j            n
#                                   =                  =             =
#                           Amplitudenoise    Componentnoise Additivenoise

#                           [ Info ECG: x3 must be scaled higher because the x1 and x2 waves
#              have a higher amplitude.
#              The following mix matrix wi represents a suitable scaling of
#              the ECG-signal:wi = [rand(2,12)-0.5;0.5+0.5*rand(1,12)]]

#     Attributes:
#         typ: (string) Type of DGL solver: BlueSky, Lorenz, Roessler, ECG
#         optional arguments:
#                'Dimension' - defines the number of dimensions (total)
#                'FixAmplitudeNoise' - SNR of the fix amplitude Noise (in dB)
#                'DimNoise' - dimensions of the component noise
#                'ComponentNoise' - SNR of the random component Noise (in dB)
#                'FixComponentNoise' - SNR of the fix component Noise (in dB)
#                'AdditiveNoise' - SNR ratio of the random additive Noise (in dB)
#                'FixAdditiveNoise' - SNR of the fix additive Noise (in dB)
#                'MixMatrix_wi' - fix or set Matrix, which gets multiplied to the
#                                 signal and the amplitudenoise
#                'RandomMixMatrix_wi' - random Matrix, which gets multiplied to the
#                                       signal and the amplitudenoise. For
#                                       activation input random number.
#                'MixMatrix_psij' - fix or set Matrix, which gets multiplied to the
#                                   componentnoise
#                'RandomMixMatrix_psij' - random Matrix, which gets multiplied to the
#                                         componentnoise. For activation
#                                         input random number.
#                'SampleRate' - Frequency of the signal, which determines
#                               the number of data points until the end of
#                               the interval.
#                'IntervLength' - Intervall length, ode45 will run for
#                                 IntervLength[time vector]
            
#              [ Info Dimension and MixMatrix_wi: Either dimension or wi can be set
#                                                 - but not together!]
#              [ Info DimNoise and MixMatrix_psij: Either dimNoise or psij can be set
#                                                 - but not together!]
#     """
        

def simulation_data(type, **kwargs):
    
    defaultDimension = 0
    defaultDimNoise = 0
    apply_AmplitudeNoise = True
    apply_ComponentNoise = True
    apply_AdditiveNoise = True
    fix_AmplitudeNoise = True
    fix_ComponentNoise = True
    fix_AdditiveNoise = True
    random_MixMatrix_wi = True
    random_MixMatrix_psij = True
    defaultMixMatrix_wi = 0
    defaultMixMatrix_psij = 0
    defaultSampleRate = 100
    defaultIntervalLength = 100

    #Accept individual/ optional inputs. If no value entered, use default value
    if 'Dimension' in kwargs:
        Dimension = kwargs['Dimension']
    else:
        Dimension = defaultDimension

    if 'AmplitudeNoise' in kwargs:
        AmplitudeNoise = kwargs['AmplitudeNoise']
    else:
        AmplitudeNoise = 0

    if 'ComponentNoise' in kwargs:
        ComponentNoise = kwargs['ComponentNoise']
    else:
        ComponentNoise = 0

    if 'AdditiveNoise' in kwargs:
        AdditiveNoise = kwargs['AdditiveNoise']
    else:
        AdditiveNoise = 0

    if 'FixAmplitudeNoise' in kwargs:
        FixAmplitudeNoise = kwargs['FixAmplitudeNoise']
    else:
        FixAmplitudeNoise = 0

    if 'FixComponentNoise' in kwargs:
        FixComponentNoise = kwargs['FixComponentNoise']
    else:
        FixComponentNoise = 0

    if 'FixAdditiveNoise' in kwargs:
        FixAdditiveNoise = kwargs['FixAdditiveNoise']
    else:
        FixAdditiveNoise = 0

    if 'DimNoise' in kwargs:
        DimNoise = kwargs['DimNoise']
    else:
        DimNoise = defaultDimNoise


    MixMatrix_wi = kwargs.get('MixMatrix_wi', defaultMixMatrix_wi)
    MixMatrix_psij = kwargs.get('MixMatrix_psij', defaultMixMatrix_psij)
    RandomMixMatrix_wi = kwargs.get('RandomMixMatrix_wi', 0)
    RandomMixMatrix_psij = kwargs.get('RandomMixMatrix_psij', 0)       
    
    if 'SampleRate' in kwargs:
        SampleRate = kwargs['SampleRate']
    else:
        SampleRate = defaultSampleRate

    if 'IntervLength' in kwargs:
        IntervLength = kwargs['IntervLength']
    else:
        IntervLength = defaultIntervalLength


    #set auxilary variables for optional inputs
    if 'AmplitudeNoise' not in kwargs:
        apply_AmplitudeNoise = False

    if 'ComponentNoise' not in kwargs:
        apply_ComponentNoise = False

    if 'AdditiveNoise' not in kwargs:
        apply_AdditiveNoise = False

    if 'FixAmplitudeNoise' not in kwargs:
        fix_AmplitudeNoise = False

    if 'FixComponentNoise' not in kwargs:
        fix_ComponentNoise = False

    if 'FixAdditiveNoise' not in kwargs:
        fix_AdditiveNoise = False

    if 'RandomMixMatrix_wi' not in kwargs:
        random_MixMatrix_wi = False

    if 'RandomMixMatrix_psij' not in kwargs:
        random_MixMatrix_psij = False

    #Print the variables which are activated and are not 
    aux_var = [apply_AmplitudeNoise, apply_ComponentNoise,apply_AdditiveNoise, fix_AmplitudeNoise, fix_ComponentNoise, fix_AdditiveNoise]
    aux_var_string = ['AmplitudeNoise', 'ComponentNoise','AdditiveNoise', 'AmplitudeNoise', 'ComponentNoise', 'AdditiveNoise']

    for i in range(0, len(aux_var)):
        if aux_var[i]:
            print(f'{aux_var_string[i]} is activated')
        else:
            print(f'{aux_var_string[i]} is not activated')

    

    k = np.arange(0, IntervLength + 1/SampleRate, 1/SampleRate)
           
    #set of the different types of DGL Solver
    if type == 'Roessler':
        a = 0.15
        b = 0.20
        c = 10

        # # Roessler attractor
        # def f_roessler(x, t):
        #     dxdt = [-x[1]-x[2],
        #             x[0]+a*x[1],
        #             b+x[2]*(x[0]-c)]
        #     return dxdt
        
        f_roessler = lambda x, t: [-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)]

        # initial values
        yinit = [5, 1, 1]
        # solver options
        odeopt = {'rtol': 0.0001, 'atol': 0.0001, 'h0': 0.5, 'hmax': 0.5}
        # solve ODE
        x_data = odeint(f_roessler, yinit, k, **odeopt).T

        

        # t_data1 is just t in this case
        t_data = k.T
        #t_data = t_data.reshape((1, t_data.shape[0]))

    else:
        raise ValueError('Unknown type of system')
    
    


    """Set dimensions and matrices"""
    DimSignal = x_data.shape[0]

    #set Mixmatrix_wi using a random orthogonal matrix
    given_MixMatrix_wi = not np.isscalar(MixMatrix_wi)

    if given_MixMatrix_wi:
        if Dimension == 0:
            Dimension = MixMatrix_wi.shape[0]
        else:
            raise ValueError('Input Dimension will be ignored, since MixMatrix_wi is specified!')
    else:
        if random_MixMatrix_wi:
            if Dimension == 0:
                Dimension = 3
                np.random.seed(seed=int(time.time())) 
                MixMatrix_wi = np.random.rand(DimSignal, Dimension) - 0.5
            else:
                Dimension = Dimension
                np.random.seed(seed=int(time.time())) 
                MixMatrix_wi = np.random.rand(DimSignal, Dimension) - 0.5
        else:
            if Dimension == 0:
                Dimension = 3
                np.random.seed(seed = 500)
                MixMatrix_wi = np.random.rand(DimSignal, Dimension) - 0.5
            else:
                Dimension = Dimension
                np.random.seed(seed = 500)
                #MixMatrix_wi = np.random.rand(Dimension, DimSignal) - 0.5 vorher - aber flasches ergebnis, da Werte falsch angeordnet in matrix - geändert für alle if statements
                MixMatrix_wi = np.random.rand(DimSignal, Dimension) - 0.5

    # Set DimNoise, ComponentNoise, and MixMatrix_psij
    given_MixMatrix_psij = not np.isscalar(MixMatrix_psij)
    
    if apply_ComponentNoise or fix_ComponentNoise:
        if given_MixMatrix_psij:
            if DimNoise != 0:
                raise ValueError('Input DimNoise will be ignored, since MixMatrix_psij is specified!')
            DimNoise = MixMatrix_psij.shape[1]
        else:
            if DimNoise != 0:
                if random_MixMatrix_psij:
                    np.random.seed(seed=int(time.time()))
                    MixMatrix_psij = np.random.rand(DimNoise, MixMatrix_wi.shape[1]) - 0.5
                else:
                    np.random.seed(600)
                    MixMatrix_psij = np.random.rand(DimNoise, MixMatrix_wi.shape[1]) - 0.5
            else:
                raise ValueError('DimNoise or MixMatrix_psij have to be specified!')
            
        # DimNoise needs to be < Dimension - DimSignal
        if DimNoise > MixMatrix_wi.shape[1] - DimSignal:
            raise ValueError('Dimension of Noise needs to be <= Dimension - DimSignal!')
    else:
        if given_MixMatrix_psij:
            raise ValueError('MixMatrix_psij cannot be set without ComponentNoise!')
        if DimNoise != 0:
            raise ValueError('DimNoise cannot be set without ComponentNoise!')  


    # ERROR request

    # Lines of MixMatrix_wi = Lines of MixMatrix_psij
    if apply_ComponentNoise or fix_ComponentNoise:
        if MixMatrix_wi.shape[1] != MixMatrix_psij.shape[1]:
            raise ValueError('Lines of MixMatrix_wi need to match lines of MixMatrix_psij!')
    
    # Set MixMatrix_wi must have DimSignal columns
    #if MixMatrix_wi.shape[1] != DimSignal:
    if MixMatrix_wi.shape[0] != DimSignal:
        raise ValueError('Numbers of MixMatrix_wi column need to be DimSignal!')
    
    # Dimension needs to be set >= DimSignal
    if MixMatrix_wi.shape[1] < DimSignal:
        raise ValueError('Dimension needs to be set >= DimSignal!')
    
    # wi needs to be linear-independent
    if given_MixMatrix_wi:
        if np.linalg.matrix_rank(MixMatrix_wi) < DimSignal:
            raise ValueError('Set matrix wi contains linear dependency! Reset the matrix in order that it is no longer linear dependent.')
    else:
        while np.linalg.matrix_rank(MixMatrix_wi) < DimSignal:
            MixMatrix_wi = np.random.rand(Dimension, DimSignal) - 0.5

    # wi and psij need to be linear-independent
    if apply_ComponentNoise or fix_ComponentNoise:
        if given_MixMatrix_psij:
            if np.linalg.matrix_rank(MixMatrix_psij) < DimNoise:
                raise ValueError('Set matrix psij contains linear dependency! Reset the matrix in order that it is no longer linear dependent.')
            else:
                pass
        else:
            pass
        N = DimSignal + DimNoise
        a = np.concatenate([MixMatrix_wi, MixMatrix_psij], axis=0)
        if given_MixMatrix_wi and given_MixMatrix_psij:
            if np.linalg.matrix_rank(a) < N:
                raise ValueError('Set matrix wi and psij contain linear dependency! Reset the matrices in order that they are no longer linear dependent.')
            else:
                pass
        elif not given_MixMatrix_wi and given_MixMatrix_psij:
            while np.linalg.matrix_rank(a) < N:
                MixMatrix_wi = np.random.rand(Dimension, DimSignal) - 0.5
                a = [MixMatrix_wi, MixMatrix_psij]
        else:
            while np.linalg.matrix_rank(a) < N:
                MixMatrix_psij = np.random.rand(MixMatrix_wi.shape[1], DimNoise)
                a = [MixMatrix_wi, MixMatrix_psij]


    ## Simulated Data
    # Data without any noise simulation

    signal = np.dot(MixMatrix_wi.T, x_data)
    #signal = np.dot(MixMatrix_wi, x_data)

    signal_mean = np.mean(np.square(signal.ravel()))
    data = signal #bis hier stimmt alles genau - aber auch nur anfangs 

    # Set AmplitudeNoise
    if apply_AmplitudeNoise:
        AmplitudeNoise = kwargs['AmplitudeNoise']
        np.random.seed(seed = int(time()))
        amp_noise = np.dot(MixMatrix_wi.transpose(), np.random.randn(*x_data.shape))
        amp_noise_mean = np.mean(np.square(amp_noise))
        scale_noise = np.sqrt(signal_mean / amp_noise_mean * (10 ** (-AmplitudeNoise / 10)))
        data += scale_noise * amp_noise

    # Set FixAmplitudeNoise
    if fix_AmplitudeNoise:
        AmplitudeNoise = kwargs['FixAmplitudeNoise']
        np.random.seed(seed = 100)
        amp_noise = np.dot(MixMatrix_wi.transpose(), np.random.randn(*x_data.shape))
        amp_noise_mean = np.mean(np.square(amp_noise))
        scale_noise = np.sqrt(signal_mean / amp_noise_mean * (10 ** (-AmplitudeNoise / 10)))
        data += scale_noise * amp_noise

    # Set ComponentNoise
    if apply_ComponentNoise:
        ComponentNoise = kwargs['ComponentNoise']
        np.random.seed(150)
        #np.random.seed( seed = int(time()) )
        com_noise = np.dot(MixMatrix_psij.transpose(), norm.ppf(np.random.rand(MixMatrix_psij.shape[0], x_data.shape[1])))
        #com_noise = np.dot(MixMatrix_psij.transpose(), np.random.randn(MixMatrix_psij.shape[0], x_data.shape[1]))
        com_noise_mean = np.mean(np.square(com_noise))
        com_scale_noise = np.sqrt(signal_mean / com_noise_mean * (10 ** (-ComponentNoise / 10)))
        data += com_scale_noise * com_noise

    # Set FixComponentNoise
    if fix_ComponentNoise:
        ComponentNoise = kwargs['FixComponentNoise']
        np.random.seed(seed = 200)
        com_noise = np.dot(MixMatrix_psij.transpose(), norm.ppf(np.random.rand(MixMatrix_psij.shape[0], x_data.shape[1])))
        #com_noise = np.dot(MixMatrix_psij.transpose(), np.random.randn(MixMatrix_psij.shape[0], x_data.shape[1]))
        com_scale_noise = np.sqrt(signal_mean / com_noise_mean * (10 ** (-ComponentNoise / 10)))
        data += com_scale_noise * com_noise

    # Set AdditiveNoise
    if apply_AdditiveNoise:
        AdditiveNoise = kwargs['AdditiveNoise']
        np.random.seed(seed = int(time()))
        add_noise = np.random.randn(*signal.shape)
        add_noise_mean = np.mean(np.square(add_noise))
        add_scale_noise = np.sqrt(signal_mean / add_noise_mean * (10 ** (-AdditiveNoise / 10)))
        data += add_scale_noise * add_noise

    # Set FixAdditiveNoise
    if fix_AdditiveNoise:
        AdditiveNoise = kwargs['FixAdditiveNoise']
        np.random.seed(seed = 300)
        add_noise = norm.ppf(np.random.rand(signal.shape[1], signal.shape[0])).T
        add_noise_mean = np.mean(np.square(add_noise))
        add_scale_noise = np.sqrt(signal_mean / add_noise_mean * (10 ** (-AdditiveNoise / 10)))
        data += add_scale_noise * add_noise

    dic = space_time_data(t_data, data)
    return dic


L = 100        # Intervall length
N = 20         # Signal dimension
n = 3
p = 4

np.random.seed(None)
MixMatrix = np.random.rand(N, n+p)
# MixMatrix = np.eye(N, n+p)
W = MixMatrix[:, 0:n]
Psi = MixMatrix[:, n:n+p]

#data, time = simulation_data('Roessler', FixComponentNoise = 0, IntervLength = L, MixMatrix_wi = W, MixMatrix_psij = Psi, FixAdditiveNoise = 20)
#data, time = simulation_data('Roessler', IntervLength = L, MixMatrix_wi = np.eye(n));
#dic = simulation_data('Roessler',Dimension = 10,IntervLength = 100, ComponentNoise = 2, FixAdditiveNoise = 3, DimNoise = 1)
#print(dic['derivateData'])
