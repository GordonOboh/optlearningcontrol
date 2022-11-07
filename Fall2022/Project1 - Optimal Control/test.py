%matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi
import quadrotor

# we can get its mass, half length (r), gravity constant
print(f'm is {quadrotor.MASS}')
print(f'r is {quadrotor.LENGTH}')
print(f'I is {quadrotor.INERTIA}')
print(f'g is {quadrotor.GRAVITY}')

# we can also get the integration step used in the simulation
print(f'dt is {quadrotor.DELTA_T}')

# we can get the size of its state and control vector
print(f'number of states {quadrotor.NUMBER_STATES} and number of controls {quadrotor.NUMBER_CONTROLS}')
print('the states are indexed as follows: x, vx, y, vy, theta, omega')

# we can simulate the robot but we need to provide a controller of the following form
def dummy_controller(state, i):
    """
        the prototype of a controller is as follows
        state is a column vector containing the state of the robot
        i is the index corresponding to the time step in the horizon (useful to index gains K for e.g.)
        
        this controller needs to return an array of size (2,)
    """

    def solve_LQR(A, B, Q, R, QN, N):
        P_n=[]
        K_n=[]
        list_of_P = []
        list_of_K = []
        K=N+1
        for i in range(0,K):
            if i == 0:
                PN=QN
                P=PN
                list_of_P=[P]

            else:
                Y=np.matmul(np.transpose(B),P)
                Y1=np.linalg.inv(np.matmul(Y,B)+R)
                K_n=np.matmul(np.matmul(-Y1,Y),A)

                Z=np.matmul(np.transpose(A),P)
                Z1=np.matmul(np.matmul(Z,B),K_n)
                P_n=Q+np.matmul(Z,A)+(Z1)
                P=P_n

                list_of_P.insert(0,P)
                list_of_K.insert(0,K_n)

        return list_of_P, list_of_K

    dt=quadrotor.DELTA_T
    mass=quadrotor.MASS
    Len=quadrotor.LENGTH
    Inertia=quadrotor.INERTIA
    grav=quadrotor.GRAVITY
    r_size=state.shape[0]

    #_____________________
    x00=1
    y00=1
    theta00=np.deg2rad(0)
    state[0]=x00
    state[2]=y00
    state[5]=0
    theta=state[5]
    #____________________


    #________________________
    A=np.eye(r_size)
    A[0,1]=dt
    A[2,3]=dt
    A[4,5]=dt
    #________________________
    B=np.zeros((r_size,2))
    B[1,0]=-sin(theta)*dt/mass
    B[3,0]=cos(theta)*dt/mass
    B[5,0]=Len*dt/Inertia
    B[1,1]=-sin(theta)*dt/mass
    B[3,1]=cos(theta)*dt/mass
    B[5,1]=-Len*dt/Inertia
    #________________________
    C=np.zeros((r_size,))
    C[3]=grav
    #______________________
    Q=np.eye(r_size)*100
    R=np.eye(2)*0.01
    QN=Q
    N=1*10*3
    
    P,K=solve_LQR(A,B,Q,R,QN,N)

    x0=state
    x=x0.reshape(len(x0),1) 
    for i in range(N):
        K_n=K[i][:][:]   #Initializing necessary variable to store K_n
        x_n = x[:,i]     #Initializing necessary variable to store x_n

        u_n=np.matmul(K_n,x_n) #Formula from Prof L. Righetti
        if i == 0: #Conditional to resolve an error, because stacking doesn't work with empty variables
            u=u_n.reshape(len(u_n),1) #rehape of u/u_n to allow for stacking
        else:
            u=np.hstack((u,u_n.reshape(len(u_n),1))) #rehape of v_n to allow for stacking

        x_n1=(A@x_n)+(B@u_n)+C #Formula from Prof L. Righetti
        #C causes the read lines, experiment later.
        x=np.hstack((x,x_n1.reshape(len(x0),1)))    #rehape of x_n to allow for stacking
    u_optimal=u

    u_star=K*state
    # here we do nothing and just return some non-zero control
    return u_star[N-1,:,5]#u_optimal[:,N-1]




# we can now simulate for a given number of time steps - here we do 10 seconds
horizon_length = 1000
z0 = np.zeros([quadrotor.NUMBER_STATES,])
t, state, u = quadrotor.simulate(z0, dummy_controller, horizon_length, disturbance = False)

# we can plot the results
plt.figure(figsize=[9,6])

plt.subplot(2,3,1)
plt.plot(t, state[0,:])
plt.legend(['X'])

plt.subplot(2,3,2)
plt.plot(t, state[2,:])
plt.legend(['Y'])

plt.subplot(2,3,3)
plt.plot(t, state[4,:])
plt.legend(["theta"])

plt.subplot(2,3,4)
plt.plot(t, state[1,:])
plt.legend(['Vx'])
plt.xlabel('Time [s]')

plt.subplot(2,3,5)
plt.plot(t, state[3,:])
plt.legend(['Vy'])
plt.xlabel('Time [s]')

plt.subplot(2,3,6)
plt.plot(t, state[5,:])
plt.legend(['omega'])
plt.xlabel('Time [s]')

# we can also plot the control
plt.figure()
plt.plot(t[:-1], u.T)
plt.legend(['u1', 'u2'])
plt.xlabel('Time [s]')

# we can also simulate with perturbations
t, state, u = quadrotor.simulate(z0, dummy_controller, horizon_length, disturbance = True)

# we can plot the results
plt.figure(figsize=[9,6])

plt.subplot(2,3,1)
plt.plot(t, state[0,:])
plt.legend(['X'])

plt.subplot(2,3,2)
plt.plot(t, state[2,:])
plt.legend(['Y'])

plt.subplot(2,3,3)
plt.plot(t, state[4,:])
plt.legend(["theta"])

plt.subplot(2,3,4)
plt.plot(t, state[1,:])
plt.legend(['Vx'])
plt.xlabel('Time [s]')

plt.subplot(2,3,5)
plt.plot(t, state[3,:])
plt.legend(['Vy'])
plt.xlabel('Time [s]')

plt.subplot(2,3,6)
plt.plot(t, state[5,:])
plt.legend(['omega'])
plt.xlabel('Time [s]')

# we can also plot the control
plt.figure()
plt.plot(t[:-1], u.T)
plt.legend(['u1', 'u2'])
plt.xlabel('Time [s]')

quadrotor.animate_robot(state,u)






















