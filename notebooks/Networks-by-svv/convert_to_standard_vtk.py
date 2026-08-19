"""
Convert an svVascularize .tree(.npz) network into a legacy ASCII VTK file
that matches the format of `56_adnr_new.vtk`.

Geometry, length and radius come straight from svVascularize.
Everything else is derived with standard 1-D hemodynamics relations so the
file carries the same fields as the reference:

    CELL_DATA  : vessel_id, a0, a_d, E, h_wall, p_d, p0, L, r_d
    POINT_DATA : boundary_id, R1, R2, C, P_out

Only the physiological constants in the CONFIG block below are assumptions;
the geometry, L, r_d, areas and IDs are exact.
"""

import numpy as np
from svv.tree.tree import Tree


# ==================================================================
# Files
# ==================================================================

input_file  = "./trees/network_10000.tree.npz"
output_file = "./trees/network_2k_adnr.vtk"

dataset_title = "svVascularize vessel network"


# ==================================================================
# CONFIG  -  physiological model parameters
# (defaults chosen )
# ==================================================================

# --- wall material -------------------------------------------------
E_YOUNG = 2.25e5          # Young's modulus  [Pa]   -> field "E"
P_D     = 1.0e4           # diastolic pressure [Pa] -> field "p_d"
P_0     = 0.0             # reference pressure [Pa] -> field "p0"

# --- wall-thickness law:  E*h/r = k1*exp(k2*r) + k3 --------
#     => h = (r / E) * (k1*exp(k2*r) + k3)
WALL_K1 = 6.538312e4
WALL_K2 = -4.615404e2
WALL_K3 = 2.632577e4

# --- RCR Windkessel at each outlet --------------------------------
#     R_total = DELTA_P / Q          (Q = svVascularize terminal flow)
#     R1 = R1_FRACTION * R_total     (characteristic impedance)
#     R2 = R_total - R1
#     C  = TAU / R_total             (fixed RC time constant)
DELTA_P     = P_D - P_0    # perfusion pressure across the outlet bed [Pa]
R1_FRACTION = 0.2
TAU         = 0.283        # RC time constant [s]
P_OUT       = 0.0          # outlet pressure  [Pa] -> field "P_out"


# ==================================================================
# svVascularize data-column layout 
# ==================================================================

COL_PROXIMAL      = slice(0, 3)
COL_DISTAL        = slice(3, 6)
COL_PARENT        = 17
COL_PROXIMAL_NODE = 18
COL_DISTAL_NODE   = 19
COL_LENGTH        = 20
COL_RADIUS        = 21
COL_FLOW          = 22


# ==================================================================
# Load tree
# ==================================================================

tree = Tree.load(input_file)
data = np.asarray(tree.data)

print("Tree loaded successfully!")
print("Number of vessel segments:", tree.segment_count)
print("Data shape:", data.shape)

proximal      = data[:, COL_PROXIMAL]
distal        = data[:, COL_DISTAL]
proximal_node = data[:, COL_PROXIMAL_NODE].astype(int)
distal_node   = data[:, COL_DISTAL_NODE].astype(int)
length        = data[:, COL_LENGTH].astype(float)
radius        = data[:, COL_RADIUS].astype(float)
flow          = np.abs(data[:, COL_FLOW].astype(float))

n_vessels = len(data)


# ==================================================================
# Unique VTK points (vessels sharing a junction share a point)
# ==================================================================

node_ids     = np.concatenate([proximal_node, distal_node])
unique_nodes = np.unique(node_ids)
node_to_point = {node_id: i for i, node_id in enumerate(unique_nodes)}

node_coordinates = {}
for i in range(n_vessels):
    node_coordinates[int(proximal_node[i])] = proximal[i]
    node_coordinates[int(distal_node[i])]   = distal[i]

points = np.array([node_coordinates[n] for n in unique_nodes])
n_points = len(points)

# line cells, remapped to point indices
cells = np.array([
    (node_to_point[int(proximal_node[i])], node_to_point[int(distal_node[i])])
    for i in range(n_vessels)
])


# ==================================================================
# CELL_DATA  (per vessel)
# ==================================================================

vessel_id = np.arange(n_vessels, dtype=float)

r_d = radius.copy()
L   = length.copy()

# a0 = a_d = pi * r_d^2
a_d = np.pi * r_d ** 2
a0  = a_d.copy()

E   = np.full(n_vessels, E_YOUNG)
p_d = np.full(n_vessels, P_D)
p0  = np.full(n_vessels, P_0)

# wall thickness (Olufsen):  h = (r/E) * (k1*exp(k2*r) + k3)
h_wall = (r_d / E_YOUNG) * (WALL_K1 * np.exp(WALL_K2 * r_d) + WALL_K3)


# ==================================================================
# POINT_DATA  (per node)  -  boundary_id + RCR Windkessel
# ==================================================================

# classify nodes by topology
is_proximal = np.isin(unique_nodes, proximal_node)   # acts as a parent
is_distal   = np.isin(unique_nodes, distal_node)     # acts as a child

boundary_id = np.full(n_points, 25555555555, dtype=int)      # interior by default

# root: never appears as a child
root_mask = ~is_distal
boundary_id[root_mask] = 0

# outlets: never appear as a parent -> number them 1, 2, 3, ...
outlet_point_idx = np.where(~is_proximal)[0]
outlet_point_idx = outlet_point_idx[np.argsort(unique_nodes[outlet_point_idx])]
for k, pidx in enumerate(outlet_point_idx, start=1):
    boundary_id[pidx] = k

# RCR values live on the outlet nodes; zero everywhere else
R1    = np.zeros(n_points)
R2    = np.zeros(n_points)
C     = np.zeros(n_points)
P_out = np.full(n_points, P_OUT)

# map each outlet node -> the vessel that terminates there (distal node)
vessel_by_distal = {int(distal_node[i]): i for i in range(n_vessels)}

for pidx in outlet_point_idx:
    node = int(unique_nodes[pidx])
    vi   = vessel_by_distal.get(node)
    if vi is None:
        continue
    q = flow[vi]
    if q <= 0.0 or not np.isfinite(q):
        continue
    R_total   = DELTA_P / q
    # dor single resistor model commentiong R1 AND C
    #R1[pidx]  = R1_FRACTION * R_total
    R1[pidx]  = 0
    R2[pidx]  = R_total - R1[pidx]
    #C[pidx]   = TAU / R_total
    C = 0


# ==================================================================
# Write legacy ASCII VTK  (byte-format matched to 56_adnr_new.vtk)
# ==================================================================

def write_scalar(f, name, values, fmt):
    f.write(f"SCALARS {name} double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in values:
        f.write(fmt.format(v) + "\n")

with open(output_file, "w") as f:

    # ---- header ----
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{dataset_title}\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # ---- points ----
    f.write(f"POINTS {n_points} double\n")
    for p in points:
        f.write(f"{p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")

    # ---- cells ----
    f.write(f"CELLS {n_vessels} {n_vessels * 3}\n")
    for a, b in cells:
        f.write(f"2 {a} {b}\n")

    # ---- cell types (VTK_LINE = 3) ----
    f.write(f"CELL_TYPES {n_vessels}\n")
    for _ in range(n_vessels):
        f.write("3\n")

    # ---- cell data ----
    f.write(f"CELL_DATA {n_vessels}\n")
    write_scalar(f, "vessel_id", vessel_id, "{:.15e}")
    write_scalar(f, "a0",        a0,        "{:.15e}")
    write_scalar(f, "a_d",       a_d,       "{:.6e}")
    write_scalar(f, "E",         E,         "{:.15e}")
    write_scalar(f, "h_wall",    h_wall,    "{:.6e}")
    write_scalar(f, "p_d",       p_d,       "{:.15e}")
    write_scalar(f, "p0",        p0,        "{:.15e}")
    write_scalar(f, "L",         L,         "{:.15e}")
    write_scalar(f, "r_d",       r_d,       "{:.15e}")

    # ---- point data ----
    f.write(f"POINT_DATA {n_points}\n")
    f.write("SCALARS boundary_id double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in boundary_id:
        f.write(f"{int(v)}\n")
    write_scalar(f, "R1",    R1,    "{:.12e}")
    write_scalar(f, "R2",    R2,    "{:.12e}")
    write_scalar(f, "C",     C,     "{:.12e}")
    write_scalar(f, "P_out", P_out, "{:.12e}")


print()
print("VTK file created successfully!")
print(output_file)
print(f"  points   : {n_points}")
print(f"  vessels  : {n_vessels}")
print(f"  outlets  : {len(outlet_point_idx)}")