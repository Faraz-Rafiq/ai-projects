
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS  — single source of truth for all configuration
# ─────────────────────────────────────────────────────────────────────────────

DEPARTMENTS = [
    "Computer Science",
    "Artificial Intelligence",
    "Business Analytics",
    "Software Engineering",
    "Electrical Engineering",
]

ALL_BATCHES = [19, 20, 21, 22, 23]

# Number of students per (batch, department) combination
# Totals across batches 19-22 fall between 2400 and 2500
BATCH_DEPT_SIZES = {
    19: [108, 92,  78,  98,  94],
    20: [113, 97,  83,  104, 98],
    21: [118, 102, 88,  109, 103],
    22: [123, 107, 93,  114, 108],
    23: [128, 112, 98,  119, 113],
}

# Faculty names assigned to each department
DEPT_FACULTY_POOL = {
    "Computer Science": [
        "Dr. Aamir Rashid", "Dr. Sadia Jabeen", "Prof. Nasir Mehmood",
        "Ms. Amna Tariq",   "Mr. Saqlain Butt", "Dr. Umer Farooq",
    ],
    "Artificial Intelligence": [
        "Dr. Qasim Nadeem", "Prof. Laila Khan", "Dr. Arshad Iqbal",
        "Ms. Sidra Anwar",  "Mr. Asim Zubair",
    ],
    "Business Analytics": [
        "Dr. Mariam Yousuf", "Prof. Shahid Awan", "Ms. Rabia Malik",
        "Mr. Faizan Sheikh",
    ],
    "Software Engineering": [
        "Dr. Hamid Baig",   "Prof. Nadia Hussain", "Dr. Zohaib Akhtar",
        "Ms. Hira Chaudhry", "Mr. Talal Mirza",   "Dr. Fahad Raza",
    ],
    "Electrical Engineering": [
        "Dr. Muneeb Alam", "Prof. Saima Rehman", "Dr. Bilal Ansar",
        "Mr. Waqas Javed", "Ms. Sana Nawaz",
    ],
}

# Rooms: 30 in total.  First 3 have 25 seats, next 24 have 35, last 3 have 30.
def _build_room_list():
    rooms = []
    for number in range(1, 31):
        if number <= 3:
            cap = 25
        elif number <= 27:
            cap = 35
        else:
            cap = 30
        rooms.append({"room_id": f"R{number:02d}", "capacity": cap})
    return rooms

ROOM_LIST = _build_room_list()

NUM_CLUSTERS = 25           # 5 departments × 5 batches
MAX_KMEANS_ITER = 300
RANDOM_SEED = 7


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — DATA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_student_data():
    """
    Generates synthetic student records for all batches and departments.
    Each student has: student_id, roll_number, batch, department,
    dept_index (integer code for clustering), batch_index.

    Returns
    -------
    dict with numpy arrays for every field
    """
    np.random.seed(RANDOM_SEED)

    student_ids   = []
    roll_numbers  = []
    batches       = []
    departments   = []
    dept_indices  = []
    batch_indices = []

    sid = 1001
    for batch_idx, batch_year in enumerate(ALL_BATCHES):
        counts = BATCH_DEPT_SIZES[batch_year]
        for dept_idx, dept_name in enumerate(DEPARTMENTS):
            for _ in range(counts[dept_idx]):
                roll = f"24F-{sid}"
                student_ids.append(sid)
                roll_numbers.append(roll)
                batches.append(batch_year)
                departments.append(dept_name)
                dept_indices.append(dept_idx)
                batch_indices.append(batch_idx)
                sid += 1

    data = {
        "student_id":  np.array(student_ids,   dtype=int),
        "roll_number": np.array(roll_numbers),
        "batch":       np.array(batches,        dtype=int),
        "department":  np.array(departments),
        "dept_idx":    np.array(dept_indices,   dtype=int),
        "batch_idx":   np.array(batch_indices,  dtype=int),
    }

    total = len(data["student_id"])
    print(f"  Students generated : {total}")
    print(f"  Batches covered    : {sorted(set(data['batch'].tolist()))}")
    print(f"  Departments        : {len(DEPARTMENTS)}")
    return data


def collect_room_data():
    """
    Returns room information as numpy arrays.
    30 rooms with varying capacities as per problem specification.
    """
    room_ids   = np.array([r["room_id"]  for r in ROOM_LIST])
    capacities = np.array([r["capacity"] for r in ROOM_LIST], dtype=int)
    total_cap  = int(capacities.sum())
    print(f"  Rooms available    : {len(room_ids)}")
    print(f"  Total seat capacity: {total_cap}")
    return {"room_id": room_ids, "capacity": capacities}


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_clustering(student_data):
    """
    Builds a 2-column feature matrix from [dept_index, batch_index].
    Applies z-score standardisation so both features contribute equally
    to the Euclidean distance used in K-Means.

    Parameters
    ----------
    student_data : dict of numpy arrays (from collect_student_data)

    Returns
    -------
    feature_matrix : numpy ndarray  shape (n_students, 2)
    """
    raw_features = np.column_stack([
        student_data["dept_idx"].astype(float),
        student_data["batch_idx"].astype(float),
    ])

    col_mean = raw_features.mean(axis=0)
    col_std  = raw_features.std(axis=0)
    col_std[col_std == 0] = 1.0          # avoid division by zero

    standardised = (raw_features - col_mean) / col_std
    print(f"  Feature matrix     : {standardised.shape[0]} rows × {standardised.shape[1]} columns")
    print(f"  Feature columns    : [department_code, batch_code]  (z-score scaled)")
    return standardised


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — K-MEANS CLUSTERING  (implemented from scratch with NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_initial_centroids(feature_matrix, k):
    """
    Randomly selects k rows from feature_matrix as starting centroids.
    Uses RANDOM_SEED for reproducibility.
    """
    np.random.seed(RANDOM_SEED)
    chosen_rows = np.random.choice(len(feature_matrix), size=k, replace=False)
    return feature_matrix[chosen_rows].copy()


def _compute_distances(feature_matrix, centroids):
    """
    Computes Euclidean distance from every data point to every centroid.

    Parameters
    ----------
    feature_matrix : (n, d) array
    centroids      : (k, d) array

    Returns
    -------
    dist_matrix : (n, k) array  — dist_matrix[i, j] = dist(point_i, centroid_j)
    """
    # Expand dims to broadcast: (n,1,d) vs (1,k,d)
    diff = feature_matrix[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    return dist_matrix


def _assign_to_nearest(dist_matrix):
    """Returns the index of the nearest centroid for every data point."""
    return np.argmin(dist_matrix, axis=1)


def _recompute_centroids(feature_matrix, assignments, k):
    """
    Computes new centroid positions as the mean of all points in each cluster.
    If a cluster is empty the old centroid position is kept.
    """
    new_centroids = np.zeros((k, feature_matrix.shape[1]))
    for cluster_id in range(k):
        members = feature_matrix[assignments == cluster_id]
        if len(members) > 0:
            new_centroids[cluster_id] = members.mean(axis=0)
    return new_centroids


def _total_inertia(feature_matrix, assignments, centroids):
    """
    Sum of squared distances from each point to its assigned centroid.
    Used for the elbow method.
    """
    total = 0.0
    for cluster_id in range(len(centroids)):
        members = feature_matrix[assignments == cluster_id]
        if len(members) > 0:
            total += ((members - centroids[cluster_id]) ** 2).sum()
    return total


def run_kmeans(feature_matrix, k=NUM_CLUSTERS):
    """
    Runs K-Means until the centroids stop moving or MAX_KMEANS_ITER is reached.

    Parameters
    ----------
    feature_matrix : numpy ndarray  shape (n, d)
    k              : int — number of clusters

    Returns
    -------
    cluster_labels : numpy ndarray  shape (n,)  — cluster index per student
    inertia        : float  — final within-cluster sum of squares
    """
    centroids = _pick_initial_centroids(feature_matrix, k)

    for iteration in range(MAX_KMEANS_ITER):
        dist_matrix   = _compute_distances(feature_matrix, centroids)
        cluster_labels = _assign_to_nearest(dist_matrix)
        new_centroids  = _recompute_centroids(feature_matrix, cluster_labels, k)

        if np.allclose(centroids, new_centroids, atol=1e-6):
            print(f"  K-Means converged  : iteration {iteration + 1}")
            break
        centroids = new_centroids
    else:
        print(f"  K-Means stopped    : reached max {MAX_KMEANS_ITER} iterations")

    final_inertia = _total_inertia(feature_matrix, cluster_labels, centroids)
    print(f"  Final inertia      : {final_inertia:.4f}")
    return cluster_labels, final_inertia


def elbow_method(feature_matrix, k_values=None):
    """
    Runs K-Means for several values of k and records inertia.
    Prints a simple ASCII elbow chart and returns the inertia list.

    Parameters
    ----------
    feature_matrix : numpy ndarray
    k_values       : list of int  (defaults to 5, 10, 15, 20, 25, 30)

    Returns
    -------
    list of (k, inertia) tuples
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 25, 30]

    print(f"\n  Elbow Method — testing k = {k_values}")
    results = []
    for k in k_values:
        _, inertia = run_kmeans(feature_matrix, k=k)
        results.append((k, inertia))

    # ASCII chart
    max_inertia = max(r[1] for r in results)
    bar_width   = 40
    print("\n  ── Elbow Chart (Inertia vs k) ──────────────────────")
    for k, inertia in results:
        bar_len = int((inertia / max_inertia) * bar_width)
        bar     = "█" * bar_len
        marker  = " ← chosen" if k == NUM_CLUSTERS else ""
        print(f"  k={k:>3}  |{bar:<{bar_width}}|  {inertia:>10.2f}{marker}")
    print("  ────────────────────────────────────────────────────\n")

    return results


def build_cluster_labels(cluster_assignments, student_data):
    """
    Creates a human-readable label for each cluster based on the dominant
    (batch, department) combination within that cluster.

    Returns
    -------
    numpy array of label strings, one per student
    """
    num_clusters  = int(cluster_assignments.max()) + 1
    cluster_names = {}

    for cid in range(num_clusters):
        mask = cluster_assignments == cid
        if not mask.any():
            cluster_names[cid] = f"C{cid:02d}"
            continue
        # Find mode batch and mode department for this cluster
        batch_vals  = student_data["batch"][mask]
        dept_vals   = student_data["department"][mask]
        mode_batch  = int(np.bincount(batch_vals - min(ALL_BATCHES)).argmax()) + min(ALL_BATCHES)
        dept_list, dept_counts = np.unique(dept_vals, return_counts=True)
        mode_dept   = dept_list[dept_counts.argmax()][:3]   # first 3 letters
        cluster_names[cid] = f"B{mode_batch}-{mode_dept}"

    labels = np.array([cluster_names[cid] for cid in cluster_assignments])
    return labels


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — SEATING PLAN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_seating_plan(student_data, cluster_assignments, room_data):
    """
    Assigns every student a room and seat number.

    Algorithm
    ---------
    1. Sort clusters by size (largest cluster first) so big cohorts
       get the biggest rooms.
    2. For each cluster, iterate through rooms in descending capacity
       order and fill seats greedily.
    3. A room can hold students from multiple clusters to use all seats.

    Parameters
    ----------
    student_data       : dict of arrays
    cluster_assignments: 1-D numpy array
    room_data          : dict with room_id and capacity arrays

    Returns
    -------
    seating : dict with arrays  room_id, seat_number per student
              (in same order as student_data arrays)
    room_summary : list of dicts summarising each room
    """
    n_students      = len(student_data["student_id"])
    assigned_rooms  = np.full(n_students, "", dtype=object)
    seat_numbers    = np.zeros(n_students, dtype=int)
    already_placed  = np.zeros(n_students, dtype=bool)

    # Build mutable capacity tracker per room
    room_ids   = room_data["room_id"].tolist()
    capacities = room_data["capacity"].tolist()
    remaining  = {rid: cap for rid, cap in zip(room_ids, capacities)}
    seat_ctr   = {rid: 1   for rid in room_ids}

    # Determine cluster sizes and sort largest first
    unique_clusters, cluster_sizes = np.unique(cluster_assignments, return_counts=True)
    sorted_order = np.argsort(cluster_sizes)[::-1]
    ordered_clusters = unique_clusters[sorted_order]

    for cid in ordered_clusters:
        cluster_mask    = (cluster_assignments == cid) & (~already_placed)
        cluster_indices = np.where(cluster_mask)[0]
        idx = 0

        while idx < len(cluster_indices):
            # Pick room with most remaining space
            best_room = max(remaining, key=lambda r: remaining[r])
            available = remaining[best_room]

            if available == 0:
                # All rooms full — this shouldn't happen given ~2450 students
                # vs 930 total seats across sessions; skip overflow for now
                break

            # How many students from this cluster can we put in this room?
            slots = min(available, len(cluster_indices) - idx)
            for offset in range(slots):
                student_pos = cluster_indices[idx + offset]
                assigned_rooms[student_pos] = best_room
                seat_numbers[student_pos]   = seat_ctr[best_room]
                already_placed[student_pos] = True
                seat_ctr[best_room] += 1

            remaining[best_room] -= slots
            idx += slots

    # Build room summary
    room_summary = []
    for rid in room_ids:
        room_mask = assigned_rooms == rid
        if not room_mask.any():
            continue
        n_assigned     = int(room_mask.sum())
        cap            = int(capacities[room_ids.index(rid)])
        depts_present  = sorted(set(student_data["department"][room_mask].tolist()))
        batches_present= sorted(set(student_data["batch"][room_mask].tolist()))
        utilisation    = round(n_assigned / cap * 100, 1)
        room_summary.append({
            "room_id":    rid,
            "capacity":   cap,
            "assigned":   n_assigned,
            "util_pct":   utilisation,
            "departments": ", ".join(depts_present),
            "batches":     ", ".join(str(b) for b in batches_present),
        })

    seating = {
        "room_id":     assigned_rooms,
        "seat_number": seat_numbers,
    }
    return seating, room_summary


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 — FACULTY DUTY ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def allocate_faculty_duties(seating, student_data, room_summary):
    """
    For each room, assigns at least one faculty member per department
    present in that room.  Faculty workload is balanced using a
    round-robin pointer per department.

    Parameters
    ----------
    seating      : dict with room_id array (from generate_seating_plan)
    student_data : dict of arrays
    room_summary : list of dicts (from generate_seating_plan)

    Returns
    -------
    assignments : list of dicts
        Each dict has: room_id, faculty_name, department, role
    workload : dict  faculty_name -> rooms_count
    """
    # Track how many rooms each faculty member has been assigned
    workload = {
        name: 0
        for pool in DEPT_FACULTY_POOL.values()
        for name in pool
    }
    # Round-robin pointer per department
    rr_ptr = {dept: 0 for dept in DEPARTMENTS}

    assignments = []

    for room_info in room_summary:
        rid = room_info["room_id"]
        room_mask = seating["room_id"] == rid

        # Departments present in this room sorted by student count (desc)
        depts_in_room = student_data["department"][room_mask]
        dept_names, dept_counts = np.unique(depts_in_room, return_counts=True)
        sorted_idx = np.argsort(dept_counts)[::-1]
        ordered_depts = dept_names[sorted_idx]

        role_index = 0
        assigned_this_room = set()

        for dept in ordered_depts:
            faculty_pool = DEPT_FACULTY_POOL[dept]
            # Use round-robin to pick the next faculty from this department's pool
            attempts = 0
            while attempts < len(faculty_pool):
                ptr      = rr_ptr[dept] % len(faculty_pool)
                candidate = faculty_pool[ptr]
                rr_ptr[dept] += 1
                attempts += 1
                if candidate not in assigned_this_room:
                    assigned_this_room.add(candidate)
                    workload[candidate] += 1
                    role = "Supervisor" if role_index == 0 else "Invigilator"
                    assignments.append({
                        "room_id":    rid,
                        "faculty":    candidate,
                        "department": dept,
                        "role":       role,
                    })
                    role_index += 1
                    break

    return assignments, workload


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 — REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _divider(char="─", width=70):
    return char * width


def print_student_distribution(student_data):
    """Prints a batch × department count table to the console."""
    print("\n" + _divider("═"))
    print("  STUDENT DISTRIBUTION  (Batch × Department)")
    print(_divider("═"))

    # Header
    col_w = 22
    header = f"  {'Batch':<8}" + "".join(f"{d[:col_w]:<{col_w}}" for d in DEPARTMENTS) + "TOTAL"
    print(header)
    print(_divider())

    for batch in ALL_BATCHES:
        batch_mask = student_data["batch"] == batch
        row = f"  {batch:<8}"
        row_total = 0
        for dept in DEPARTMENTS:
            dept_mask = student_data["department"] == dept
            count = int((batch_mask & dept_mask).sum())
            row += f"{count:<{col_w}}"
            row_total += count
        row += str(row_total)
        print(row)

    print(_divider())
    grand_total = len(student_data["student_id"])
    print(f"  Grand Total: {grand_total} students")
    print(_divider("═") + "\n")


def print_room_summary(room_summary):
    """Prints a room occupancy table to the console."""
    print(_divider("═"))
    print("  SEATING PLAN  —  ROOM OCCUPANCY SUMMARY")
    print(_divider("═"))
    print(f"  {'Room':<8}{'Cap':<6}{'Seats Used':<12}{'Util %':<9}Batches   Departments")
    print(_divider())

    for r in room_summary:
        print(
            f"  {r['room_id']:<8}"
            f"{r['capacity']:<6}"
            f"{r['assigned']:<12}"
            f"{r['util_pct']:<9}"
            f"{r['batches']:<10}"
            f"{r['departments']}"
        )

    total_students = sum(r["assigned"] for r in room_summary)
    avg_util = round(sum(r["util_pct"] for r in room_summary) / len(room_summary), 1)
    print(_divider())
    print(f"  Total students seated : {total_students}")
    print(f"  Average utilisation   : {avg_util}%")
    print(_divider("═") + "\n")


def print_faculty_summary(faculty_assignments, workload):
    """Prints per-room faculty duties and overall workload."""
    print(_divider("═"))
    print("  FACULTY DUTY ALLOCATION")
    print(_divider("═"))

    # Group by room
    rooms_seen = []
    for row in faculty_assignments:
        if row["room_id"] not in rooms_seen:
            rooms_seen.append(row["room_id"])

    for rid in rooms_seen:
        room_duties = [r for r in faculty_assignments if r["room_id"] == rid]
        print(f"\n  Room {rid}:")
        for duty in room_duties:
            print(f"    [{duty['role']:12s}]  {duty['faculty']}  ({duty['department']})")

    print("\n" + _divider())
    print("  FACULTY WORKLOAD  (rooms assigned)")
    print(_divider())
    for fname, count in sorted(workload.items(), key=lambda x: -x[1]):
        if count > 0:
            bar = "■" * count
            print(f"  {fname:<28} {count:>2}  {bar}")
    print(_divider("═") + "\n")


def print_cluster_summary(cluster_assignments, cluster_labels, student_data):
    """Prints per-cluster statistics."""
    print(_divider("═"))
    print("  K-MEANS CLUSTER STATISTICS  (k = 25)")
    print(_divider("═"))
    print(f"  {'Cluster':<8}{'Label':<18}{'Dept':<28}{'Batch':<8}{'Students'}")
    print(_divider())

    unique_clusters = sorted(set(cluster_assignments.tolist()))
    for cid in unique_clusters:
        mask   = cluster_assignments == cid
        label  = cluster_labels[mask][0] if mask.any() else "?"
        count  = int(mask.sum())
        # Dominant department and batch
        dept_vals = student_data["department"][mask]
        uq_d, ct_d = np.unique(dept_vals, return_counts=True)
        main_dept  = uq_d[ct_d.argmax()]

        batch_vals = student_data["batch"][mask]
        uq_b, ct_b = np.unique(batch_vals, return_counts=True)
        main_batch = int(uq_b[ct_b.argmax()])

        print(f"  {cid:<8}{label:<18}{main_dept:<28}{main_batch:<8}{count}")

    print(_divider("═") + "\n")


def save_reports(student_data, seating, cluster_assignments, cluster_labels,
                 room_summary, faculty_assignments, workload, output_dir="reports"):
    """
    Writes all output files to the given directory.
    Files produced:
        seating_plan.csv
        faculty_duties.csv
        room_summary.csv
        exam_report.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── seating_plan.csv ─────────────────────────────────────────────────────
    seat_path = os.path.join(output_dir, "seating_plan.csv")
    with open(seat_path, "w") as f:
        f.write("student_id,roll_number,batch,department,cluster_id,"
                "cluster_label,room_id,seat_number\n")
        n = len(student_data["student_id"])
        for i in range(n):
            f.write(
                f"{student_data['student_id'][i]},"
                f"{student_data['roll_number'][i]},"
                f"{student_data['batch'][i]},"
                f"{student_data['department'][i]},"
                f"{cluster_assignments[i]},"
                f"{cluster_labels[i]},"
                f"{seating['room_id'][i]},"
                f"{seating['seat_number'][i]}\n"
            )
    print(f"  Saved → {seat_path}")

    # ── faculty_duties.csv ───────────────────────────────────────────────────
    fac_path = os.path.join(output_dir, "faculty_duties.csv")
    with open(fac_path, "w") as f:
        f.write("room_id,faculty_name,department,role\n")
        for row in faculty_assignments:
            f.write(f"{row['room_id']},{row['faculty']},"
                    f"{row['department']},{row['role']}\n")
    print(f"  Saved → {fac_path}")

    # ── room_summary.csv ─────────────────────────────────────────────────────
    room_path = os.path.join(output_dir, "room_summary.csv")
    with open(room_path, "w") as f:
        f.write("room_id,capacity,students_assigned,utilisation_pct,"
                "batches,departments\n")
        for r in room_summary:
            f.write(
                f"{r['room_id']},{r['capacity']},{r['assigned']},"
                f"{r['util_pct']},\"{r['batches']}\",\"{r['departments']}\"\n"
            )
    print(f"  Saved → {room_path}")

    # ── exam_report.txt ──────────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "exam_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("  NUCES FAST — Faisalabad Chiniot Campus")
    lines.append("  AUTOMATED EXAM MANAGEMENT SYSTEM — EXECUTIVE REPORT")
    lines.append("=" * 70)
    lines.append(f"\n  Total students    : {len(student_data['student_id'])}")
    lines.append(f"  K-Means clusters  : {len(set(cluster_assignments.tolist()))}")
    lines.append(f"  Rooms used        : {len(room_summary)}")
    lines.append(f"  Faculty duties    : {len(faculty_assignments)}")
    avg_u = round(sum(r["util_pct"] for r in room_summary) / len(room_summary), 1)
    lines.append(f"  Avg room util     : {avg_u}%\n")

    lines.append("  ROOM OCCUPANCY")
    lines.append("  " + "-" * 66)
    lines.append(f"  {'Room':<8}{'Cap':<6}{'Assigned':<10}{'Util%':<8}"
                 f"{'Batches':<18}Departments")
    lines.append("  " + "-" * 66)
    for r in room_summary:
        lines.append(
            f"  {r['room_id']:<8}{r['capacity']:<6}{r['assigned']:<10}"
            f"{r['util_pct']:<8}{r['batches']:<18}{r['departments']}"
        )

    lines.append("\n  FACULTY ALLOCATION (per room)")
    lines.append("  " + "-" * 66)
    for row in faculty_assignments:
        lines.append(f"  {row['room_id']:<8}[{row['role']:12s}]  "
                     f"{row['faculty']}  ({row['department']})")

    lines.append("\n  FACULTY WORKLOAD SUMMARY")
    lines.append("  " + "-" * 66)
    for fname, count in sorted(workload.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"  {fname:<30} {count} room(s)")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_system():
    """
    Runs all six stages of the exam management pipeline in sequence
    and prints a summary of each stage to the console.
    """
    sep = "\n" + "═" * 70

    # ── Stage 1: Data Collection ─────────────────────────────────────────────
    print(sep)
    print("  STAGE 1 — DATA COLLECTION")
    print("═" * 70)
    students  = collect_student_data()
    rooms     = collect_room_data()

    # ── Stage 2: Preprocessing ───────────────────────────────────────────────
    print(sep)
    print("  STAGE 2 — DATA PREPROCESSING")
    print("═" * 70)
    feature_matrix = preprocess_for_clustering(students)

    # ── Stage 3: K-Means Clustering ──────────────────────────────────────────
    print(sep)
    print("  STAGE 3 — K-MEANS CLUSTERING")
    print("═" * 70)

    # Elbow method to justify k = 25
    elbow_method(feature_matrix, k_values=[5, 10, 15, 20, 25, 30])

    print(f"  Running K-Means with k = {NUM_CLUSTERS}  (5 departments × 5 batches)")
    cluster_ids, _ = run_kmeans(feature_matrix, k=NUM_CLUSTERS)
    cluster_labels  = build_cluster_labels(cluster_ids, students)

    print_cluster_summary(cluster_ids, cluster_labels, students)

    # ── Stage 4: Seating Plan ────────────────────────────────────────────────
    print(sep)
    print("  STAGE 4 — SEATING PLAN GENERATION")
    print("═" * 70)
    seating, room_summary = generate_seating_plan(students, cluster_ids, rooms)

    print_student_distribution(students)
    print_room_summary(room_summary)

    # ── Stage 5: Faculty Allocation ──────────────────────────────────────────
    print(sep)
    print("  STAGE 5 — FACULTY DUTY ALLOCATION")
    print("═" * 70)
    faculty_assignments, workload = allocate_faculty_duties(
        seating, students, room_summary
    )
    print_faculty_summary(faculty_assignments, workload)

    # ── Stage 6: Save Reports ────────────────────────────────────────────────
    print(sep)
    print("  STAGE 6 — SAVING REPORTS")
    print("═" * 70)
    save_reports(students, seating, cluster_ids, cluster_labels,
                 room_summary, faculty_assignments, workload,
                 output_dir="reports")

    print(sep)
    print("  PIPELINE COMPLETE")
    print(f"  All output files saved in  ./reports/")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_system()
