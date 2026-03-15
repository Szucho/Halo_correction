import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

"""
script to process the "nrhos" folder containing a family.csv and a trajectories.csv
Bertalan Szuchovszky 15.03.2026.

Pipeline:
    - read the csv files via pandas (+ get metadata: mu, L1_x, L2_x from families.csv)
    - calculate perilune & apolune distances (in km using DU = 384400 km)
    - calculate Jacobi constant and use it to create a color map
    - figure of the initial orbit in xy, xz, yz planes
    - figure of ALL the orbit trajectories in xy, xz, yz planes
    - 3D figure of ALL the orbit trajectories
    - figure of the stability indices as a function of perilune radius 
        -> check Zhang Renyong: A review of periodic orbits in the crtbp Fig 16
    - figure of maximum Lyapunov exponents as a function of perilune radius

Output: figures

Dependencies: pandas, matplotlib, numpy, the existence of "nrhos" folder at the right place (outside scripts folder)

Note: I used figures like these in my BSc thesis for the L1 & L2 cislunar NRHO families :D
      Important finding: Lyapunov exponents will show a suspicios, almost exponential looking
                         monotone increase as the perilune radius shrinks believed to be caused 
                         purely by numerical errors of the integrator. I understand that
                         those orbits are physicly not possible and would collide with the Moon
                         BUT thinking about trajectories for landing this also highlights a weakness
                         of this whole process; without regularization one should not start designing
                         orbits for Moon landing without regularizing the crtbp. Regularization is
                         also strongly recommended before one starts checking chaotic indicators 
"""

def Jacobi_C(mu, y0):
    x,y,z,vx,vy,vz = y0

    r1 = np.sqrt((x-mu)**2+y**2+z**2)
    r2 = np.sqrt((x+1-mu)**2+y**2+z**2)
    
    U = (1-mu)/r1 + mu/r2 + 0.5*((1-mu)*((x-mu)**2+y**2) + mu*((x+1-mu)**2+y**2))
    v_sq = vx**2+vy**2+vz**2

    return 2*U-v_sq


def perilune_km(orbit_group, mu, DU=384400.0):
    x  = orbit_group["x"].values
    y  = orbit_group["y"].values
    z  = orbit_group["z"].values
    r  = np.sqrt((x + 1 - mu)**2 + y**2 + z**2)
    return r.min() * DU

def apolune_km(orbit_group, mu, DU=384400.0):
    x  = orbit_group["x"].values
    y  = orbit_group["y"].values
    z  = orbit_group["z"].values
    r  = np.sqrt((x + 1 - mu)**2 + y**2 + z**2)
    return r.max() * DU


if __name__=="__main__":
    family = pd.read_csv("./nrhos/family.csv", comment="#")
#firs line is the comment line holding metadata (mu, lagrange points) for the family
    with open("./nrhos/family.csv") as f:
        meta_line = f.readline().strip("# \n")  # "mu=... L1_x=... L2_x=..."

    params = dict(item.split("=") for item in meta_line.split())
    mu  = float(params["mu"])
    L1x = float(params["L1_x"])
    L2x = float(params["L2_x"])

    P1 = mu
    P2 = mu-1

    traj = pd.read_csv("./nrhos/trajectories.csv")

    orbit0 = traj[traj["orbit_id"]==0]

    perilune = {}
    apolune  = {}
    for oid, grp in traj.groupby("orbit_id"):
        perilune[oid] = perilune_km(grp, mu)
        apolune[oid]  = apolune_km(grp, mu)


    #::::::::::::..
    #plots

    #1st orbit - corrected from our guess
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.plot(orbit0["x"], orbit0["y"])
    ax1.scatter(P2, 0, c="k", s=15, label="Moon")
    ax1.scatter(L1x, 0, c="r", marker="x", label=r"L$_1$")
    ax1.scatter(L2x, 0, c="g", marker="^", label=r"L$_2$")
    ax1.grid(ls=":", alpha=0.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    ax2.plot(orbit0["x"], orbit0["z"])
    ax2.grid(ls=":", alpha=0.5)
    ax2.scatter(P2, 0, c="k", s=15, label="Moon")
    ax2.scatter(L1x, 0, c="r", marker="x", label=r"L$_1$")
    ax2.scatter(L2x, 0, c="g", marker="^", label=r"L$_2$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.legend()

    ax3.plot(orbit0["y"], orbit0["z"])
    ax3.grid(ls=":", alpha=0.5)
    ax3.scatter(0,0, c="k", s=15, label="Moon")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("./nrhos/figs/orbit0.png", bbox_inches="tight", dpi=300)
    plt.show()



    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.scatter(P2, 0, c="k", s=15, label="Moon")
    ax1.scatter(L1x, 0, c="r", marker="x", label=r"L$_1$")
    ax1.scatter(L2x, 0, c="g", marker="^", label=r"L$_2$")
    ax1.grid(ls=":", alpha=0.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()

    ax2.grid(ls=":", alpha=0.5)
    ax2.scatter(P2, 0, c="k", s=15, label="Moon")
    ax2.scatter(L1x, 0, c="r", marker="x", label=r"L$_1$")
    ax2.scatter(L2x, 0, c="g", marker="^", label=r"L$_2$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.legend()

    ax3.grid(ls=":", alpha=0.5)
    ax3.scatter(0,0, c="k", s=15, label="Moon")
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    ax3.legend()

    #loop over all orbits generated via pseudo-arclength method
    family["JC"] = family.apply(lambda row: Jacobi_C(mu, [row.x0, row.y0, row.z0, row.vx0, row.vy0, row.vz0]), axis=1)

    norm = mcolors.Normalize(vmin=family["JC"].min(), vmax=family["JC"].max())
    cmap = plt.get_cmap("plasma")

    for oid, grp in traj.groupby("orbit_id"):
        x, y, z    = grp["x"].values, grp["y"].values, grp["z"].values
        vx, vy, vz = grp["vx"].values, grp["vy"].values, grp["vz"].values

        #Jacobi constant will be used for coloring
        color = cmap(norm(family.loc[oid,"JC"]))
        ax1.plot(grp["x"], grp["y"], color = color)
        ax2.plot(grp["x"], grp["z"], color = color)
        ax3.plot(grp["y"], grp["z"], color = color)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3, label="Jacobi constant C")
    plt.tight_layout()
    plt.savefig("./nrhos/figs/all_trajectories.png", bbox_inches="tight", dpi=300)
    plt.show()

    
    #3D plot of the family
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(P2, 0, 0, s=20, c="k", label="Moon")
    ax.scatter(L1x, 0, c="r", marker="x", label=r"L$_1$")
    ax.scatter(L2x, 0, c="g", marker="^", label=r"L$_2$")
    for oid, grp in traj.groupby("orbit_id"):
        x, y, z    = grp["x"].values, grp["y"].values, grp["z"].values
        vx, vy, vz = grp["vx"].values, grp["vy"].values, grp["vz"].values

        #Jacobi constant will be used for coloring
        color = cmap(norm(family.loc[oid,"JC"]))
        ax.plot3D(grp["x"], grp["y"], grp["z"], color = color)

    ax.axis("equal")
    ax.grid(ls=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("./nrhos/figs/3D_trajectories.png", bbox_inches="tight", dpi=300)
    plt.show()


    #stability indices based on apolune distance 
    fig, ax = plt.subplots()
    for i, (oid, row) in enumerate(family.iterrows()):
        nu1, nu2 = row.nu1, row.nu2
        r = perilune[oid]
        color = cmap(norm(family.loc[oid,"JC"]))
        label1 = r"$\nu_1$" if i == 0 else None
        label2 = r"$\nu_2$" if i == 0 else None
        ax.scatter(r, nu1, color=color, marker="o", label=label1)
        ax.scatter(r, nu2, color=color, marker="x", label=label2)

    ax.grid(ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Perilune distance [km]")
    ax.set_ylabel("Stability index")
    ax.legend()
    plt.tight_layout()
    plt.savefig("./nrhos/figs/stability_indices.png", bbox_inches="tight", dpi=300)
    plt.show()


    #lyapunov index based on apolune distance
    fig, ax = plt.subplots()
    for i, (oid, row) in enumerate(family.iterrows()):
        expo = row.lambda_max
        r = perilune[oid]
        color = cmap(norm(family.loc[oid, "JC"]))
        label = r"$\lambda_{max}$" if i==0 else None
        ax.scatter(r, expo, color=color, marker=".", label=label)

    ax.grid(ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Perilune distance [km]")
    ax.set_ylabel("Lyapunov exponent")
    ax.legend()
    plt.tight_layout()
    plt.savefig("./nrhos/figs/lyapunov_expos.png", bbox_inches="tight", dpi=300)
    plt.show()
