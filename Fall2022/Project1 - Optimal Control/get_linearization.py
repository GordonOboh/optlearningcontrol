import numpy as np
from sympy.utilities.lambdify import lambdify


from sympy import exp, sqrt, sin, cos, Matrix
from sympy import symbols, Symbol
import quadrotor

N=30

# state(z)=[x,V_x,y,V_y,theta,omega]
x,V_x,y,V_y,theta,omega,u1,u2 = symbols('x V_x y V_y theta omega u1 u2')
def get_linearization(z,u):
    # Description
    #z = Matrix([[x],[V_x],[y],[V_y],[theta],[omega]])
    #u = Matrix([[u1],[u2]])
    dt=quadrotor.DELTA_T
    mass=quadrotor.MASS
    length=quadrotor.LENGTH
    Ine=quadrotor.INERTIA
    grav=quadrotor.GRAVITY
    #state=Matrix
    #_______________________
    x     = z[0]
    V_x   = z[1]
    y     = z[2]
    V_y   = z[3]
    theta = z[4]
    omega = z[5]

    u1 = u[0]
    u2 = u[1]
    #________________________


    #__________________________________1
    # Continuous Dynamics for Quadrotor in Symbolic
    A_pseudo = Matrix([[V_x],[0],[V_y],[-grav],[omega],[0]])
    B_pseudo = Matrix([[0],[(-sin(theta)/mass)*(u1+u2)],[0],[(cos(theta)/mass)*(u1+u2)],[0],[(length/Ine)*(u1-u2)]])
    #C_pseudo=   
    dzdt=A_pseudo+B_pseudo
    dzdt_T=dzdt.T
    #f_cont = lambdify((x,V_x,y,V_y,theta,omega,u1,u2),dzdt_T,'numpy')
    #___________________________________1

    #___________________________________2
    # Discrete Dynamics for Quadrotor in Symbolic
    z_n_T= z.T
    z_next = z_n_T + (dt*dzdt_T)
    #f_nx = lambdify((x,V_x,y,V_y,theta,omega,u1,u2),z_next,'numpy')
    #___________________________________2

    #___________________________________3
    # Jacobian from the Discrete Dynamics in Symbolic and Numeric
    A_sym = z_next.jacobian(z)
    B_sym = z_next.jacobian(u)
    print(B_sym)
    A_num = lambdify((x,V_x,y,V_y,theta,omega,u1,u2),A_sym,'numpy')
    B_num = lambdify((x,V_x,y,V_y,theta,omega,u1,u2),B_sym,'numpy')
    #____________________________________3

    return A_num,B_num

def inf_LQR(A, B, Q, R, QN, N):
    #N=30
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
            BT=np.transpose(B)
            AT=np.transpose(A)
            #Y=np.matmul(np.transpose(B),P)
            #Y1=np.linalg.inv(np.matmul(Y,B)+R)
            #K_n=np.matmul(np.matmul(-Y1,Y),A)
            Y = BT @ P
            Y1 = np.linalg.inv(Y @ B+R)
            K_n = -Y1 @ Y @ A

            #Z=np.matmul(np.transpose(A),P)
            #Z1=np.matmul(np.matmul(Z,B),K_n)
            #P_n=Q+np.matmul(Z,A)+(Z1)
            #P=P_n
            Z = AT @ P
            Z1 = Z @ B @ K_n
            P_n = Q + (Z @ A) + Z1
            P = P_n
            
            list_of_P.insert(0,P)
            list_of_K.insert(0,K_n)
    
    return list_of_P, list_of_K

def optimal_state_and_control(state_first,N,K):
    x=state_first.reshape(len(state_first),1)  #change the shape of x0 to allow for stacking
    for i in range(N):
        K_n=K[i][:][:]   #Initializing necessary variable to store K_n
        x_n = x[:,i]     #Initializing necessary variable to store x_n

        u_n=(K_n @ x_n) #Formula from Prof L. Righetti
        if i == 0: #Conditional to resolve an error, because stacking doesn't work with empty variables
            u=u_n.reshape(len(u_n),1) #rehape of u/u_n to allow for stacking
        else:
            u=np.hstack((u,u_n.reshape(len(u_n),1))) #rehape of u_n to allow for stacking

        x_n1=(A @ x_n)+(B @ u_n)  #Formula from Prof L. Righetti
        x=np.hstack((x,x_n1.reshape(len(x0),1)))    #rehape of x_n to allow for stacking
    state_optimal=x
    control_optimal=u
    return state_optimal,control_optimal

u11,u22 = symbols('u11 u22')

z_n = Matrix([[x],[V_x],[y],[V_y],[theta],[omega]])
u_n = Matrix([[u1],[u2]])

A,B=get_linearization(z_n,u_n)
print(A,B)
print(A(0,0,0,0,0,0,u11,u22),B(0,0,0,0,0,0,3,3))

A1=A(0,0,0,0,0,0,0,0)
B1=B


Q=np.eye(6)*100
R=np.eye(2)*0.01
QN=Q
P,K=inf_LQR(A1,B1,Q,R,QN,N)
print(K) 

#z0=
#z,u=optimal_state_and_control(z0,N,K)










