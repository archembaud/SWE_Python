# sw.py
# Dr. Matthew Smith, Swinburne University of Technology
# Solve the 2D Shallow Water Equations using Numpy arrays
# and the SHLL method.
# Slow versions of key functions are also provided for
# performance comparison.

import time
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# ========== Function declarations ============

def Compute_State_Slow(P, U, NX, NY):
	# Compute primitives P from U using for loops.
	for x in range(NX):
		for y in range(NY):
			P[x,y,0] = U[x,y,0]		# Water Height
			P[x,y,1] = U[x,y,1]/U[x,y,0]	# X vel
			P[x,y,2] = U[x,y,2]/U[x,y,0]	# Y vel
	return P


def Compute_State(P, U):
        # Compute Primitives P from U using vectors
        P[:,:,0] = U[:,:,0]   		# Water Height
        P[:,:,1] = U[:,:,1]/U[:,:,0]	# X vel
        P[:,:,2] = U[:,:,2]/U[:,:,0]	# Y vel
        return P

def Compute_Change_of_State(dU, FP, FM, NX, NY, PHI, direction):
        # Compute the change of state based on gradients of F
        FL = np.ndarray([NX,NY,3])
        FR = np.ndarray([NX,NY,3])

        if (direction == 0):
                # This is the X direction computation
                for x in range(NX):
                        if (x == 0):
                                # Left boundary - use Neuman condition
                                FL[x,:,:] = FP[x,:,:]
                                FR[x,:,:] = FM[x+1,:,:]
                        elif (x == (NX-1)):
                                # Right boundary - use Neuman as well
                                FL[x,:,:] = FP[x-1,:,:]
                                FR[x,:,:] = FM[x,:,:]
                        else:
                             	# Internal cell. All is fine and dandy.
                                FL[x,:,:] = FP[x-1,:,:]
                                FR[x,:,:] = FM[x+1,:,:]

                # Now to apply them to update the state
                dU = dU - PHI*(FP - FM + FR - FL);

        elif (direction == 1):
                # This is for the Y direction
                for y in range(NY):
                        if (y == 0):
                                # Left boundary - use Neuman
                                FL[:,y,:] = FP[:,y,:]
                                FR[:,y,:] = FM[:,y+1,:]
                        elif (y == (NY-1)):
                                # Right boundary - use Neuman
                                FL[:,y,:] = FP[:,y-1,:]
                                FR[:,y,:] = FM[:,y,:]
                        else:
                             	# Internal cell.
                                FL[:,y,:] = FP[:,y-1,:]
                                FR[:,y,:] = FM[:,y+1,:]

                # Now to apply them to update the state
                dU = dU - PHI*(FP - FM + FR - FL);

        return dU


def Compute_Fluxes_Slow(P, NX, NY, G, D, direction):
	# Compute split SHLL fluxes in each direction using slow loops
        FP = np.ndarray([NX, NY, 3])
        FM = np.ndarray([NX, NY, 3])
        U = np.ndarray([NX, NY, 3])

        # Compute fluxes in particular direction as required
        F = np.ndarray([NX,NY,3])

        # Recompute a local copy of U based on P
	for x in range(NX):
		for y in range(NY):
		        U[x,y,0] = P[x,y,0]
        		U[x,y,1] = P[x,y,0]*P[x,y,1]
        		U[x,y,2] = P[x,y,0]*P[x,y,2]

        if (direction == 0):
                # X direction.
		for x in range(NX):
			for y in range(NY):
                		a = np.sqrt(P[x,y,0]*G)
				vel = P[x,y,1]  # X velocity
                		Fr = vel/a  # Froud number
                		# Compute some important constants
                		Z1 = 0.5*(D*Fr+1.0)
                		Z2 = 0.5*D*a*(1.0-Fr*Fr)
                		Z3 = 0.5*(D*Fr-1.0)

                		# Compute X direction fluxes
                		F[x,y,0] = U[x,y,1]   # Height * X velocity
                		F[x,y,1] = U[x,y,1]*P[x,y,1] + 0.5*G*P[x,y,0]*P[x,y,0]
                		F[x,y,2] = U[x,y,1]*P[x,y,2] # Height*Xvel*Yvel

		                # Compute the SHLL forward and backward fluxes
		                FP[x,y,0] = F[x,y,0]*Z1 + U[x,y,0]*Z2
                		FP[x,y,1] = F[x,y,1]*Z1 + U[x,y,1]*Z2
                		FP[x,y,2] = F[x,y,2]*Z1 + U[x,y,2]*Z2

                		FM[x,y,0] = -F[x,y,0]*Z3 - U[x,y,0]*Z2
                		FM[x,y,1] = -F[x,y,1]*Z3 - U[x,y,1]*Z2
                		FM[x,y,2] = -F[x,y,2]*Z3 - U[x,y,2]*Z2

        elif (direction == 1):
                # Y direction
		for x in range(NX):
			for y in range(NY):
				a = np.sqrt(P[x,y,0]*G)
                		vel = P[x,y,2]  # Y Velocity
                		Fr = vel/a
                		# Compute some important constants
                		Z1 = 0.5*(D*Fr+1.0)
                		Z2 = 0.5*D*a*(1.0-Fr*Fr)
                		Z3 = 0.5*(D*Fr-1.0)

                		# Compute Y direction fluxes
                		F[x,y,0] = U[x,y,2]   # Height * Y velocity
                		F[x,y,1] = U[x,y,2]*P[x,y,1] # Height*Yvel*Xvel
                		F[x,y,2] = U[x,y,2]*P[x,y,2] + 0.5*G*P[x,y,0]*P[x,y,0]

               			# Compute the SHLL forward and backward fluxes
                		FP[x,y,0] = F[x,y,0]*Z1 + U[x,y,0]*Z2
                		FP[x,y,1] = F[x,y,1]*Z1 + U[x,y,1]*Z2
                		FP[x,y,2] = F[x,y,2]*Z1 + U[x,y,2]*Z2

                		FM[x,y,0] = -F[x,y,0]*Z3 - U[x,y,0]*Z2
                		FM[x,y,1] = -F[x,y,1]*Z3 - U[x,y,1]*Z2
                		FM[x,y,2] = -F[x,y,2]*Z3 - U[x,y,2]*Z2


	return FP, FM





def Compute_Fluxes(P, NX, NY, G, D, direction):
	# Compute split SHLL fluxes in each direction
        FP = np.ndarray([NX, NY, 3])
        FM = np.ndarray([NX, NY, 3])
        U = np.ndarray([NX, NY, 3])

        # Compute fluxes in particular direction as required
        F = np.ndarray([NX,NY,3])

        # Recompute a local copy of U based on P
        U[:,:,0] = P[:,:,0]
        U[:,:,1] = P[:,:,0]*P[:,:,1]
        U[:,:,2] = P[:,:,0]*P[:,:,2]

        # Use the direction to access the correct velocity information
        a = np.sqrt(P[:,:,0]*G)

        if (direction == 0):
                # X direction.
                # Attempt to vectorize the computation
                vel = P[:,:,1]  # X velocity
                Fr = vel/a  # Froud number
                # Compute some important constants
                Z1 = 0.5*(D*Fr+1.0)
                Z2 = 0.5*D*a*(1.0-Fr*Fr)
                Z3 = 0.5*(D*Fr-1.0)

                # Compute X direction fluxes
                F[:,:,0] = U[:,:,1]   # Height * X velocity
                F[:,:,1] = U[:,:,1]*P[:,:,1] + 0.5*G*P[:,:,0]*P[:,:,0]
                F[:,:,2] = U[:,:,1]*P[:,:,2] # Height*Xvel*Yvel

                # Compute the SHLL forward and backward fluxes
                FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
                FP[:,:,1] = F[:,:,1]*Z1 + U[:,:,1]*Z2
                FP[:,:,2] = F[:,:,2]*Z1 + U[:,:,2]*Z2

                FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
                FM[:,:,1] = -F[:,:,1]*Z3 - U[:,:,1]*Z2
                FM[:,:,2] = -F[:,:,2]*Z3 - U[:,:,2]*Z2

        elif (direction == 1):
                # Y direction
                vel = P[:,:,2]  # Y Velocity
                Fr = vel/a
                # Compute some important constants
                Z1 = 0.5*(D*Fr+1.0)
                Z2 = 0.5*D*a*(1.0-Fr*Fr)
                Z3 = 0.5*(D*Fr-1.0)

                # Compute Y direction fluxes
                F[:,:,0] = U[:,:,2]   # Height * Y velocity
                F[:,:,1] = U[:,:,2]*P[:,:,1] # Height*Yvel*Xvel
                F[:,:,2] = U[:,:,2]*P[:,:,2] + 0.5*G*P[:,:,0]*P[:,:,0]

                # Compute the SHLL forward and backward fluxes
                FP[:,:,0] = F[:,:,0]*Z1 + U[:,:,0]*Z2
                FP[:,:,1] = F[:,:,1]*Z1 + U[:,:,1]*Z2
                FP[:,:,2] = F[:,:,2]*Z1 + U[:,:,2]*Z2

                FM[:,:,0] = -F[:,:,0]*Z3 - U[:,:,0]*Z2
                FM[:,:,1] = -F[:,:,1]*Z3 - U[:,:,1]*Z2
                FM[:,:,2] = -F[:,:,2]*Z3 - U[:,:,2]*Z2

        return FP, FM

def Plot_Surface(Data,DX,DY,NX,NY):
        # Create a surface plot of the 2D data Data
        L = NX*DX
        H = NY*DY
        X,Y = np.mgrid[0:L:DX, 0:H:DY]
        Z = Data
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax.set_title('Spatial variation of Data')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

        return 0


def Plot_All_Surfaces(P, DX, DY, NX, NY):
        # Plot all results within a single figure
        L = NX*DX
        H = NY*DY
        X,Y = np.mgrid[0:L:DX, 0:H:DY]
        Z = P[:,:,0] # Height
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        fig,ax = plt.subplots(1,3,figsize=(30, 10),subplot_kw=dict(projection='3d'))
        ax[0].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[0].set_title('Height')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        # X Velocity
        Z = P[:,:,1] # X Vel
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax[1].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[1].set_title('X Velocity')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        # Y Velocity
        Z = P[:,:,2] # Y Vel
        levels = MaxNLocator(nbins=25).tick_values(Z.min(), Z.max())
        cmap = plt.get_cmap('gist_rainbow')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax[2].plot_surface(X,Y,Z, cmap=cmap, norm=norm)
        ax[2].set_title('Y Velocity')
        ax[2].set_xlabel('X')
        ax[2].set_ylabel('Y')
        plt.show()

        return 0


def Init(U, P, NX, NY):
        # Compute initial values of U and P
	for x in range(NX):
                for y in range(NY):
                        # Compute initial conditions based on location
                        if ((x < 0.6*NX) and (x > 0.4*NX) and (y < 0.6*NY) and (y > 0.4*NY)):
                                P[x,y,0] = 10.0   # Water height
                        else:
                             	P[x,y,0] = 1.0

                        # The water is initially stationary
                        P[x,y,1] = 0.0  # X speed
                        P[x,y,2] = 0.0  # Y speed

        # Now to compute U from P
        U[:,:,0] = P[:,:,0]  	# Water Height is conservative
        U[:,:,1] = P[:,:,0]*P[:,:,1] # Height*X Vel = X Mom
        U[:,:,2] = P[:,:,0]*P[:,:,2] # Y Mom
        return U, P


# Main Code section
# =================

# Solution Parameters
L = 1.0    # Length of region (x direction), m
H = 1.0    # Height of region (y direction), m
NX = 50	   # Number of cells in X direction
NY = 50    # Number of cells in Y direction
DX = (L/NX) # Grid size in X direction
DY = (H/NY) # "      "     Y direction
DT = 0.00005 # Time step, in seconds (s). Static for now.
NO_STEPS = 400 # Number of time steps to take
G = 9.81        # Gravitational accelerationm m/s2
D = 0.9		# Anti-dissipation coefficient
USE_FAST = False		# Decide which routines to use - fast (True) or not (False)

# Create our nd arrays
P = np.ndarray([NX, NY, 3])   # Primitives - height, water speed in x,y directions
U = np.ndarray([NX, NY, 3])   # Conservatives - height and momentum in x,y directions
FP = np.ndarray([NX, NY, 3])  # Forward (positive, P) fluxes of conserved quantities
FM = np.ndarray([NX, NY, 3])  # Backward (minus, M) fluxes
dU = np.ndarray([NX, NY, 3])  # Changes to conserved quantities

# Initialize the flow field
U,P = Init(U,P,NX,NY)

# Run the solver
tic = time.time()
for step in range(NO_STEPS):
	print("Time step %d of %d" % (step, NO_STEPS))

	# Reset dU
	dU = 0*dU

	# Calculate fluxes in X direction (direction = 0)
	if (USE_FAST):
		# Employ vectorization to compute fluxes
		FP, FM = Compute_Fluxes(P, NX, NY, G, D, 0)
	else:
		# Compute fluxes slowly
		FP, FM = Compute_Fluxes_Slow(P, NX, NY, G, D, 0)
	# Update dU
	dU = Compute_Change_of_State(dU, FP, FM, NX, NY, (DT/DX), 0)

	# Update fluxes in Y direction (direction = 1)
	if (USE_FAST):
		# Employ vectorization to compute fluxes
		FP, FM = Compute_Fluxes(P, NX, NY, G, D, 1)
	else:
		# Compute the fluxes slowly
		FP, FM = Compute_Fluxes_Slow(P, NX, NY, G, D, 1)
	# Update dU
	dU = Compute_Change_of_State(dU, FP, FM, NX, NY, (DT/DY), 1)

	# Update dU
	U = U + dU

	# Update our state P
	if (USE_FAST):
		P = Compute_State(P, U)
	else:
		# Use the slow method
		P = Compute_State_Slow(P, U, NX, NY)

	# End of main transient time loop segment

# Simulation is completed now. Examine the results.
toc = time.time()
Elapsed_Time = toc - tic
print("Elapsed time = %g seconds" % Elapsed_Time)
# Plot FP
Plot_All_Surfaces(P,DX,DY,NX,NY)

