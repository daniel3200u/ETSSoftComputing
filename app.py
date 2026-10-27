import ast
import heapq
import random
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Optimasi Rute Gudang (GA & NN) - Z=1",
    page_icon="📦",
    layout="wide",
)

DATA_DIR = (
    Path(__file__).resolve().parent
    / "Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company"
)

GITHUB_BASE = (
    "https://raw.githubusercontent.com/daniel3200u/ETSSoftComputing/refs/heads/main/"
    "Order%20Picking%20Dataset%20from%20a%20Warehouse%20of%20a%20Footwear%20Manufacturing%20Company"
)


def parse_point(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"Invalid support point coordinate: {value!r}")
    return pd.Series(value, index=["X", "Y", "Z"])


@st.cache_data(show_spinner="Memuat dataset gudang & membangun graf lorong...")
def load_warehouse_data():
    storage_path = DATA_DIR / "Storage_Location.csv"
    wave_path = DATA_DIR / "Picking_Wave.csv"
    support_path = DATA_DIR / "Support_Points_Navigation.csv"

    df_storage = pd.read_csv(
        storage_path if storage_path.exists() else f"{GITHUB_BASE}/Storage_Location.csv"
    )
    df_wave = pd.read_csv(
        wave_path if wave_path.exists() else f"{GITHUB_BASE}/Picking_Wave.csv",
        sep=";",
    )
    df_support = pd.read_csv(
        support_path
        if support_path.exists()
        else f"{GITHUB_BASE}/Support_Points_Navigation.csv",
        sep=";",
    )

    df_wave["locations"] = df_wave["locations"].astype(str).str.strip()
    df_storage["originalLocation"] = (
        df_storage["originalLocation"].astype(str).str.strip()
    )

    parsed = df_support["points_specified"].apply(parse_point)
    for col in ["X", "Y", "Z"]:
        df_support[col] = pd.to_numeric(parsed[col], errors="coerce")

    # Filter Z == 1 sesuai ketentuan
    df_storage_z1 = df_storage[df_storage["z"] == 1].reset_index(drop=True)
    df_storage_z1["rack_block"] = df_storage_z1["originalLocation"].apply(
        lambda s: "-".join(s.split("-")[:2])
    )

    mask = df_wave["locations"].isin(df_storage_z1["originalLocation"])
    df_wave_z1 = df_wave[mask].reset_index(drop=True)

    # Bangun Graf Lorong seperti di Notebook Soft_Computing_Test (1).ipynb
    nav = {}
    for _, row in df_support.iterrows():
        nav[str(row["labels"]).strip()] = (float(row["X"]), float(row["Y"]))

    labels = list(nav.keys())
    idx_map = {l: i for i, l in enumerate(labels)}

    xs = {}
    for l in labels:
        prefix = l.split("-")[0]
        if prefix not in xs:
            xs[prefix] = nav[l][0]

    n = len(labels)
    adj = [[] for _ in range(n)]

    def edge(a, b):
        d = abs(nav[a][0] - nav[b][0]) + abs(nav[a][1] - nav[b][1])
        adj[idx_map[a]].append((idx_map[b], d))
        adj[idx_map[b]].append((idx_map[a], d))

    for g in ("LC", "CC", "RC"):
        ls = sorted([l for l in labels if l.startswith(g)], key=lambda l: nav[l][1])
        for a, b in zip(ls, ls[1:]):
            edge(a, b)

    ys = {}
    for l in labels:
        ys.setdefault(nav[l][1], {})[l[:2]] = l
    for y_val, d in ys.items():
        if "LC" in d and "CC" in d:
            edge(d["LC"], d["CC"])
        if "CC" in d and "RC" in d:
            edge(d["CC"], d["RC"])

    # All-pairs shortest path (Dijkstra)
    d_matrix = np.full((n, n), np.inf)
    for s in range(n):
        d_matrix[s, s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > d_matrix[s, u]:
                continue
            for v, w in adj[u]:
                if d + w < d_matrix[s, v]:
                    d_matrix[s, v] = d + w
                    heapq.heappush(pq, (d + w, v))

    aisle_y = sorted({nav[l][1] for l in labels})
    nav_at = {(l[:2], nav[l][1]): l for l in labels}

    return {
        "df_storage": df_storage_z1,
        "df_wave": df_wave_z1,
        "df_support": df_support,
        "NAV": nav,
        "LABELS": labels,
        "IDX": idx_map,
        "XS": xs,
        "ADJ": adj,
        "D_MATRIX": d_matrix,
        "AISLE_Y": aisle_y,
        "NAV_AT": nav_at,
    }


def access_point(x, y, wh):
    ya = min(wh["AISLE_Y"], key=lambda a: abs(a - y))
    if x < wh["XS"]["CC"]:
        g1, g2 = "LC", "CC"
    else:
        g1, g2 = "CC", "RC"
    return ya, 0.0, wh["NAV_AT"][(g1, ya)], wh["NAV_AT"][(g2, ya)]


def get_distance_matrix(locs, depot_loc, wh):
    all_points = np.vstack([locs[:, :2], depot_loc[:2]])
    n = len(all_points)
    acc = []
    for x, y in all_points:
        ya, off, a, b = access_point(x, y, wh)
        acc.append(
            (
                x,
                ya,
                off,
                wh["IDX"][a],
                wh["IDX"][b],
                wh["NAV"][a][0],
                wh["NAV"][b][0],
            )
        )

    m_dist = np.zeros((n, n))
    d_matrix = wh["D_MATRIX"]
    for i in range(n):
        xi, yi, oi, ai, bi, axi, bxi = acc[i]
        for j in range(i + 1, n):
            xj, yj, oj, aj, bj, axj, bxj = acc[j]
            if yi == yj and axi == axj:
                d = oi + oj + abs(xi - xj)
            else:
                d = np.inf
                for ei, exi in ((ai, axi), (bi, bxi)):
                    for ej, exj in ((aj, axj), (bj, bxj)):
                        d = min(d, abs(xi - exi) + d_matrix[ei, ej] + abs(xj - exj))
                d += oi + oj
            m_dist[i, j] = m_dist[j, i] = d
    return m_dist


def hitung_total_jarak(urutan_rute, m_dist, depot_idx):
    if len(urutan_rute) == 0:
        return 0.0
    jarak_total = 0.0
    jarak_total += m_dist[urutan_rute[0], depot_idx]
    for i in range(len(urutan_rute) - 1):
        jarak_total += m_dist[urutan_rute[i], urutan_rute[i + 1]]
    jarak_total += m_dist[urutan_rute[-1], depot_idx]
    return jarak_total


def inisialisasi_populasi(ukuran_populasi, jumlah_gen):
    populasi = []
    for _ in range(ukuran_populasi):
        individu = list(range(jumlah_gen))
        random.shuffle(individu)
        populasi.append(individu)
    return populasi


def seleksi_turnamen(populasi, jarak_populasi, ukuran_turnamen=3):
    k = min(ukuran_turnamen, len(populasi))
    terpilih = random.sample(range(len(populasi)), k)
    terbaik_idx = min(terpilih, key=lambda idx: jarak_populasi[idx])
    return populasi[terbaik_idx]


def crossover_order(induk1, induk2):
    ukuran = len(induk1)
    if ukuran < 2:
        return induk1[:]
    titik1, titik2 = sorted(random.sample(range(ukuran), 2))
    anak = [-1] * ukuran
    anak[titik1:titik2] = induk1[titik1:titik2]
    posisi = titik2
    for gen in induk2:
        if gen not in anak:
            if posisi >= ukuran:
                posisi = 0
            anak[posisi] = gen
            posisi += 1
    return anak


def mutasi_swap(individu, laju_mutasi):
    if len(individu) < 2:
        return individu
    for i in range(len(individu)):
        if random.random() < laju_mutasi:
            j = random.randint(0, len(individu) - 1)
            individu[i], individu[j] = individu[j], individu[i]
    return individu


def jalankan_ga(
    jumlah_titik,
    m_dist,
    generasi_maks=150,
    ukuran_populasi=50,
    laju_mutasi=0.1,
    ukuran_turnamen=3,
):
    if jumlah_titik == 1:
        jarak = hitung_total_jarak([0], m_dist, 1)
        return [0], jarak, [jarak] * generasi_maks, []

    populasi = inisialisasi_populasi(ukuran_populasi, jumlah_titik)
    jarak_terbaik_global = float("inf")
    rute_terbaik_global = None
    riwayat_jarak = []
    log_generasi = []

    for generasi in range(generasi_maks):
        jarak_populasi = [
            hitung_total_jarak(ind, m_dist, jumlah_titik) for ind in populasi
        ]
        min_jarak = min(jarak_populasi)
        if min_jarak < jarak_terbaik_global:
            jarak_terbaik_global = min_jarak
            rute_terbaik_global = populasi[jarak_populasi.index(min_jarak)][:]
        riwayat_jarak.append(jarak_terbaik_global)

        populasi_baru = []
        for _ in range(ukuran_populasi):
            induk1 = seleksi_turnamen(populasi, jarak_populasi, ukuran_turnamen)
            induk2 = seleksi_turnamen(populasi, jarak_populasi, ukuran_turnamen)
            anak = crossover_order(induk1, induk2)
            anak = mutasi_swap(anak, laju_mutasi)
            populasi_baru.append(anak)
        populasi = populasi_baru

        if (generasi + 1) % 20 == 0 or generasi == 0 or generasi == generasi_maks - 1:
            log_generasi.append(
                {
                    "Generasi": generasi + 1,
                    "Jarak Terbaik Sementara (m)": round(jarak_terbaik_global, 2),
                    "Rata-rata Populasi (m)": round(float(np.mean(jarak_populasi)), 2),
                }
            )

    return rute_terbaik_global, jarak_terbaik_global, riwayat_jarak, log_generasi


def jalankan_nn(depot_idx, jumlah_lokasi, m_dist):
    titik_belum_dikunjungi = list(range(jumlah_lokasi))
    rute_nn = []
    total_jarak_nn = 0.0
    posisi_sekarang_idx = depot_idx

    while titik_belum_dikunjungi:
        jarak_terdekat = float("inf")
        titik_selanjutnya_idx = -1
        indeks_dalam_list = -1

        for i, idx in enumerate(titik_belum_dikunjungi):
            jarak = m_dist[posisi_sekarang_idx, idx]
            if jarak < jarak_terdekat:
                jarak_terdekat = jarak
                titik_selanjutnya_idx = idx
                indeks_dalam_list = i

        rute_nn.append(titik_selanjutnya_idx)
        total_jarak_nn += jarak_terdekat
        posisi_sekarang_idx = titik_selanjutnya_idx
        titik_belum_dikunjungi.pop(indeks_dalam_list)

    total_jarak_nn += m_dist[posisi_sekarang_idx, depot_idx]
    return rute_nn, total_jarak_nn


def get_path(p1, p2, wh):
    x1, y1 = p1
    x2, y2 = p2

    ya1, _, a1, b1 = access_point(x1, y1, wh)
    ya2, _, a2, b2 = access_point(x2, y2, wh)

    nav = wh["NAV"]
    idx_map = wh["IDX"]
    d_matrix = wh["D_MATRIX"]
    adj = wh["ADJ"]
    labels = wh["LABELS"]

    axi, bxi = nav[a1][0], nav[b1][0]
    axj, bxj = nav[a2][0], nav[b2][0]

    if ya1 == ya2 and axi == axj:
        return [(x1, y1), (x1, ya1), (x2, ya1), (x2, y2)]

    best_d = float("inf")
    best_path = []

    for ei, exi in ((a1, axi), (b1, bxi)):
        for ej, exj in ((a2, axj), (b2, bxj)):
            d = abs(x1 - exi) + d_matrix[idx_map[ei], idx_map[ej]] + abs(x2 - exj)
            if d < best_d:
                best_d = d
                dist, prev, pq = {ei: 0.0}, {}, [(0.0, ei)]
                while pq:
                    du, u = heapq.heappop(pq)
                    if u == ej:
                        break
                    if du > dist.get(u, float("inf")):
                        continue
                    for v_idx, w in adj[idx_map[u]]:
                        v = labels[v_idx]
                        if du + w < dist.get(v, float("inf")):
                            dist[v] = du + w
                            prev[v] = u
                            heapq.heappush(pq, (du + w, v))
                if ej not in dist:
                    continue
                p = [ej]
                while p[-1] != ei:
                    p.append(prev[p[-1]])
                p = p[::-1]

                path = [(x1, y1), (x1, ya1), (exi, ya1)]
                for node in p:
                    if nav[node] not in path:
                        path.append(nav[node])
                if (exj, ya2) not in path:
                    path.append((exj, ya2))
                path.append((x2, ya2))
                path.append((x2, y2))
                best_path = path
    return best_path


def buat_grafik_konvergensi(riwayat, jarak_terbaik_ga):
    fig, ax = plt.subplots(figsize=(10, 5))
    generasi = np.arange(1, len(riwayat) + 1)
    ax.plot(
        generasi,
        riwayat,
        color="#1f77b4",
        lw=2,
        label="Jarak Terbaik (Best Distance)",
    )
    ax.axhline(
        y=jarak_terbaik_ga,
        color="red",
        linestyle="--",
        label=f"Jarak Minimum Akhir ({jarak_terbaik_ga:.2f} m)",
    )
    ax.set_title(
        "Grafik Konvergensi Genetic Algorithm (GA)", fontsize=15, weight="bold"
    )
    ax.set_xlabel("Generasi", fontsize=12)
    ax.set_ylabel("Total Jarak (meter)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


def buat_visualisasi_rute(
    rute_terbaik,
    nama_algoritma,
    jarak_terbaik,
    depot,
    titik_kunjungan,
    df_rute,
    wh,
    label_skenario="Custom Rak",
):
    koordinat_rute_lengkap = (
        [depot] + [titik_kunjungan[idx] for idx in rute_terbaik] + [depot]
    )
    df_pos_floor = wh["df_storage"]
    df_sup_floor = wh["df_support"][wh["df_support"]["Z"] == 1.0]

    fig_width, fig_height = 30, 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Gambar blok rak lantai Z = 1
    for _, row in df_pos_floor.iterrows():
        x, y = row["x"], row["y"]
        block_width = 15
        xmin = x - block_width / 2
        ymin = y - 15
        rect = plt.Rectangle(
            (xmin, ymin),
            block_width,
            30,
            edgecolor="black",
            facecolor="white",
            alpha=0.5,
            linewidth=1,
        )
        ax.add_patch(rect)

    # Highlight rak yang dipilih
    for _, row in df_rute.iterrows():
        x, y = row["x"], row["y"]
        block_width = 15
        xmin = x - block_width / 2
        ymin = y - 15
        rect_sel = plt.Rectangle(
            (xmin, ymin),
            block_width,
            30,
            edgecolor="darkred",
            facecolor="#ffcccc",
            alpha=0.85,
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect_sel)

    # Support Points & Garis Lorong
    if not df_sup_floor.empty:
        support_x = df_sup_floor["X"]
        support_y = df_sup_floor["Y"]
        support_labels = df_sup_floor["labels"]
        ax.scatter(
            support_x,
            support_y,
            color="lightgray",
            s=50,
            label="Support Points",
            zorder=2,
        )
        for i, label in enumerate(support_labels):
            ax.text(
                support_x.iloc[i] + 5,
                support_y.iloc[i],
                label,
                fontsize=10,
                color="gray",
            )
        for y_val in df_sup_floor["Y"].unique():
            points_on_y = df_sup_floor[df_sup_floor["Y"] == y_val].sort_values(by="X")
            if len(points_on_y) > 1:
                ax.plot(
                    points_on_y["X"],
                    points_on_y["Y"],
                    color="lightgray",
                    linestyle="--",
                    linewidth=1,
                    zorder=1,
                )
        for x_val in df_sup_floor["X"].unique():
            points_on_x = df_sup_floor[df_sup_floor["X"] == x_val].sort_values(by="Y")
            if len(points_on_x) > 1:
                ax.plot(
                    points_on_x["X"],
                    points_on_x["Y"],
                    color="lightgray",
                    linestyle="--",
                    linewidth=1,
                    zorder=1,
                )

    # Titik Awal (DEPOT)
    ax.scatter(
        depot[0], depot[1], color="green", s=400, marker="*", zorder=5, label="DEPOT"
    )
    ax.text(
        depot[0],
        depot[1] + 20,
        "DEPOT (Start/End)",
        color="green",
        weight="bold",
        fontsize=12,
        ha="center",
        bbox=dict(
            facecolor="white", edgecolor="green", boxstyle="round,pad=0.2", alpha=0.8
        ),
    )

    # Garis Rute & Panah Arah Perjalanan
    for i in range(len(koordinat_rute_lengkap) - 1):
        p1 = koordinat_rute_lengkap[i]
        p2 = koordinat_rute_lengkap[i + 1]
        path = get_path((p1[0], p1[1]), (p2[0], p2[1]), wh)
        if not path:
            path = [(p1[0], p1[1]), (p2[0], p1[1]), (p2[0], p2[1])]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="red", ls="-", lw=2.5, alpha=0.8, zorder=3)
        for k in range(len(path) - 1):
            if path[k] == path[k + 1]:
                continue
            mid_x = (path[k][0] + path[k + 1][0]) / 2
            mid_y = (path[k][1] + path[k + 1][1]) / 2
            dx = path[k + 1][0] - path[k][0]
            dy = path[k + 1][1] - path[k][1]
            length = np.hypot(dx, dy)
            if length > 20:
                dx /= length
                dy /= length
                ax.annotate(
                    "",
                    xy=(mid_x + dx * 5, mid_y + dy * 5),
                    xytext=(mid_x - dx * 5, mid_y - dy * 5),
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.5, head_length=0.7",
                        color="red",
                        lw=2.5,
                        alpha=0.8,
                    ),
                    zorder=4,
                )

    # Label Step Kunjungan
    rute_nama_lokasi = df_rute.iloc[rute_terbaik]["displayLocation"].tolist()
    for step_idx, p in enumerate(koordinat_rute_lengkap):
        if 0 < step_idx < len(koordinat_rute_lengkap) - 1:
            nama_rak = rute_nama_lokasi[step_idx - 1]
            ax.text(
                p[0],
                p[1] + 25,
                f"Step {step_idx}\n({nama_rak})",
                color="white",
                fontsize=10,
                weight="bold",
                ha="center",
                bbox=dict(
                    facecolor="red",
                    edgecolor="darkred",
                    boxstyle="round,pad=0.3",
                    alpha=0.9,
                ),
                zorder=5,
            )

    ax.set_xticks([])
    ax.set_yticks([])

    margin = 100
    ax.set_xlim(df_pos_floor["x"].min() - margin, df_pos_floor["x"].max() + margin)
    ax.set_ylim(df_pos_floor["y"].min() - margin, df_pos_floor["y"].max() + margin)

    ax.set_title(
        f"Visualisasi Rute ({nama_algoritma}) via Lorong | {label_skenario} | Z=1\n"
        f"Total Jarak = {jarak_terbaik:.2f} m",
        fontsize=18,
        weight="bold",
    )

    red_line = mlines.Line2D(
        [],
        [],
        color="red",
        marker=">",
        linestyle="-",
        lw=2.5,
        label="Rute Kunjungan (Lorong)",
    )
    depot_marker = mlines.Line2D(
        [],
        [],
        color="green",
        marker="*",
        linestyle="None",
        markersize=15,
        label="Depot / Titik Awal",
    )
    ax.legend(handles=[red_line, depot_marker], loc="upper right", fontsize=12)
    fig.tight_layout()
    return fig


def resolve_selected_racks(selected_codes, df_storage):
    """
    Mengubah daftar kode rak yang dipilih user (bisa berupa kode blok 2 bagian
    seperti 'K-13', 'H-10' maupun kode spesifik 3 bagian seperti 'K-13-11')
    menjadi DataFrame koordinat (x, y, z=1).
    """
    rows = []
    for code in selected_codes:
        code_clean = str(code).strip().upper()
        exact = df_storage[df_storage["originalLocation"].str.upper() == code_clean]
        if not exact.empty:
            r = exact.iloc[0]
            rows.append(
                {
                    "displayLocation": code_clean,
                    "originalLocation": r["originalLocation"],
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                }
            )
            continue

        # Jika memilih kode blok rak 2-bagian (mis. K-13 atau H-10), ambil titik tengah (-12)
        block_matches = df_storage[df_storage["rack_block"].str.upper() == code_clean]
        if not block_matches.empty:
            mid = block_matches[
                block_matches["originalLocation"].str.endswith("-12")
            ]
            r = mid.iloc[0] if not mid.empty else block_matches.iloc[0]
            rows.append(
                {
                    "displayLocation": f"{code_clean} ({r['originalLocation']})",
                    "originalLocation": r["originalLocation"],
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "z": float(r["z"]),
                }
            )
    return pd.DataFrame(rows)


def main():
    st.title("📦 Optimasi Rute Pengambilan Barang pada Gudang (Z = 1)")
    st.markdown(
        "Aplikasi GUI berbasis **Genetic Algorithm (GA)** & **Nearest Neighbor (NN)** "
        "untuk optimasi rute *order picking* melalui graf lorong gudang pada lantai **Z = 1** "
        "(sesuai notebook `Soft_Computing_Test (1).ipynb`)."
    )

    wh = load_warehouse_data()
    df_storage = wh["df_storage"]
    df_wave = wh["df_wave"]
    df_support = wh["df_support"]

    # Sidebar: Pengaturan Depot & Parameter Algoritma
    with st.sidebar:
        st.header("⚙️ Konfigurasi Algoritma")

        df_sup_z1 = df_support[df_support["Z"] == 1.0].reset_index(drop=True)
        default_depot_idx = 26 if len(df_sup_z1) > 26 else 0
        depot_labels = [
            f"{row['labels']} ({int(row['X'])}, {int(row['Y'])}, Z={int(row['Z'])})"
            for _, row in df_sup_z1.iterrows()
        ]
        selected_depot_str = st.selectbox(
            "Titik Awal / Depot (Z = 1):",
            options=depot_labels,
            index=default_depot_idx,
            help="Default: CC-08 (indeks 26 pada Support_Points_Navigation.csv)",
        )
        depot_idx_in_sup = depot_labels.index(selected_depot_str)
        depot_row = df_sup_z1.iloc[depot_idx_in_sup]
        depot = np.array([float(depot_row["X"]), float(depot_row["Y"]), float(depot_row["Z"])])

        st.divider()
        st.subheader("🧬 Parameter Genetic Algorithm")
        generasi_maks = st.slider("Jumlah Generasi Maksimum", 20, 500, 150, step=10)
        ukuran_populasi = st.slider("Ukuran Populasi", 10, 200, 50, step=10)
        laju_mutasi = st.slider("Laju Mutasi (Mutation Rate)", 0.01, 0.50, 0.10, step=0.01)
        ukuran_turnamen = st.slider("Ukuran Turnamen Seleksi", 2, 10, 3, step=1)
        use_seed = st.checkbox("Gunakan Random Seed tetap (Reproducible)", value=False)
        if use_seed:
            seed_val = st.number_input("Random Seed", value=42, step=1)
        else:
            seed_val = None

    # Bagian Pemilihan Rak (Z = 1)
    st.subheader("1️⃣ Pilih Rak Penyimpanan yang Akan Dikunjungi (Lantai Z = 1)")

    mode_input = st.radio(
        "Mode Pemilihan Rak:",
        options=[
            "Pilih Rak Manual (Blok Rak seperti H-10, K-13 atau Sub-Rak Spesifik seperti K-13-11)",
            "Pilih Berdasarkan Picking Wave (Dataset Picking_Wave.csv)",
        ],
        horizontal=True,
    )

    rack_blocks = sorted(df_storage["rack_block"].unique().tolist())
    exact_locations = df_storage["originalLocation"].unique().tolist()

    if mode_input.startswith("Pilih Rak Manual"):
        col_filter, col_preset = st.columns([2, 1])
        with col_filter:
            tipe_opsi = st.selectbox(
                "Format Daftar Rak pada Pilihan:",
                options=[
                    "Gabungan: Kode Blok Rak (mis. H-10, K-13) & Lokasi Spesifik (mis. K-13-11)",
                    "Kode Blok Rak Saja (mis. A-14, H-10, K-13, M-13)",
                    "Kode Lokasi Spesifik Saja (mis. A-14-11, H-10-12, K-13-11)",
                ],
            )
        with col_preset:
            st.write("")
            st.write("")
            if st.button("📌 Muat Contoh Rak Wave 33168 (9 Lokasi)"):
                wave_33168_locs = (
                    df_wave[df_wave["waveNumber"] == 33168]["locations"]
                    .unique()
                    .tolist()
                )
                st.session_state["manual_racks"] = wave_33168_locs

        if tipe_opsi.startswith("Kode Blok Rak Saja"):
            available_options = rack_blocks
            default_selection = ["H-10", "K-13", "M-13", "P-09"]
        elif tipe_opsi.startswith("Kode Lokasi Spesifik Saja"):
            available_options = exact_locations
            default_selection = (
                df_wave[df_wave["waveNumber"] == 33168]["locations"]
                .unique()
                .tolist()
            )
        else:
            available_options = rack_blocks + exact_locations
            default_selection = ["H-10", "K-13", "M-13-11", "P-09-13"]

        if "manual_racks" not in st.session_state:
            st.session_state["manual_racks"] = [
                r for r in default_selection if r in available_options
            ]

        # Pastikan nilai session_state ada di available_options
        valid_current = [
            r for r in st.session_state["manual_racks"] if r in available_options
        ]

        selected_racks = st.multiselect(
            "Ketik dan pilih rak satu per satu sesuai urutan yang diinginkan (Z = 1):",
            options=available_options,
            default=valid_current,
            help="Ketik kode rak (contoh: H-10, K-13, atau K-13-11) lalu tekan Enter. Urutan pemilihan Anda dicatat sebagai Rute Awal.",
        )
        st.session_state["manual_racks"] = selected_racks
        label_skenario = f"Custom ({len(selected_racks)} Rak)"

    else:
        wave_counts = (
            df_wave.groupby("waveNumber")["locations"]
            .nunique()
            .reset_index()
            .rename(columns={"locations": "jumlah_lokasi"})
        )
        wave_list = wave_counts["waveNumber"].tolist()
        default_wave_idx = wave_list.index(33168) if 33168 in wave_list else 0

        selected_wave = st.selectbox(
            "Pilih Nomor Picking Wave (Z = 1):",
            options=wave_list,
            index=default_wave_idx,
            format_func=lambda w: f"Wave {w} ({wave_counts.loc[wave_counts['waveNumber'] == w, 'jumlah_lokasi'].values[0]} lokasi unik)",
        )
        selected_racks = (
            df_wave[df_wave["waveNumber"] == selected_wave]["locations"]
            .unique()
            .tolist()
        )
        label_skenario = f"Wave {selected_wave}"

    df_rute = resolve_selected_racks(selected_racks, df_storage)

    if df_rute.empty:
        st.warning("⚠️ Silakan pilih minimal 1 rak pada lantai Z = 1 terlebih dahulu.")
        return

    with st.expander(
        f"📋 Lihat Daftar & Koordinat {len(df_rute)} Rak Terpilih (Urutan Input)",
        expanded=True,
    ):
        df_preview = df_rute.copy()
        df_preview.index = np.arange(1, len(df_preview) + 1)
        df_preview.index.name = "Urutan Pilih"
        st.dataframe(
            df_preview[["displayLocation", "originalLocation", "x", "y", "z"]].rename(
                columns={
                    "displayLocation": "Rak Dipilih",
                    "originalLocation": "Kode Storage_Location (Z=1)",
                    "x": "Koordinat X (m)",
                    "y": "Koordinat Y (m)",
                    "z": "Lantai (Z)",
                }
            ),
            use_container_width=True,
        )

    # Tombol RUN
    run_clicked = st.button(
        "🚀 RUN Optimasi Genetic Algorithm & Visualisasi Rute",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        if seed_val is not None:
            random.seed(int(seed_val))
            np.random.seed(int(seed_val))

        titik_kunjungan = df_rute[["x", "y", "z"]].values
        jumlah_titik = len(titik_kunjungan)

        with st.spinner("Menghitung matriks jarak lorong & menjalankan Genetic Algorithm..."):
            m_dist = get_distance_matrix(titik_kunjungan, depot[:2], wh)

            rute_awal = list(range(jumlah_titik))
            jarak_awal = hitung_total_jarak(rute_awal, m_dist, jumlah_titik)

            rute_terbaik_ga, jarak_terbaik_ga, riwayat, log_gen = jalankan_ga(
                jumlah_titik=jumlah_titik,
                m_dist=m_dist,
                generasi_maks=generasi_maks,
                ukuran_populasi=ukuran_populasi,
                laju_mutasi=laju_mutasi,
                ukuran_turnamen=ukuran_turnamen,
            )

            rute_terbaik_nn, jarak_terbaik_nn = jalankan_nn(
                depot_idx=jumlah_titik,
                jumlah_lokasi=jumlah_titik,
                m_dist=m_dist,
            )

            st.session_state["hasil_optimasi"] = {
                "rute_terbaik_ga": rute_terbaik_ga,
                "jarak_terbaik_ga": jarak_terbaik_ga,
                "riwayat": riwayat,
                "log_gen": log_gen,
                "rute_terbaik_nn": rute_terbaik_nn,
                "jarak_terbaik_nn": jarak_terbaik_nn,
                "rute_awal": rute_awal,
                "jarak_awal": jarak_awal,
                "depot": depot,
                "depot_label": depot_row["labels"],
                "titik_kunjungan": titik_kunjungan,
                "df_rute": df_rute,
                "label_skenario": label_skenario,
            }

    # Tampilkan Hasil jika sudah di-Run
    if "hasil_optimasi" in st.session_state:
        res = st.session_state["hasil_optimasi"]
        rute_ga = res["rute_terbaik_ga"]
        jarak_ga = res["jarak_terbaik_ga"]
        riwayat = res["riwayat"]
        log_gen = res["log_gen"]
        rute_nn = res["rute_terbaik_nn"]
        jarak_nn = res["jarak_terbaik_nn"]
        rute_awal = res["rute_awal"]
        jarak_awal = res["jarak_awal"]
        depot_res = res["depot"]
        titik_res = res["titik_kunjungan"]
        df_rute_res = res["df_rute"]
        skenario_res = res["label_skenario"]

        st.divider()
        st.subheader("2️⃣ Ringkasan Hasil Optimasi Jarak Tempuh")

        m1, m2, m3, m4 = st.columns(4)
        selisih_awal = jarak_awal - jarak_ga
        persen_hemat = (selisih_awal / jarak_awal * 100) if jarak_awal > 0 else 0.0
        m1.metric(
            "🏆 Total Jarak GA (Terbaik)",
            f"{jarak_ga:.2f} m",
            delta=f"-{selisih_awal:.2f} m ({persen_hemat:.1f}% vs Awal)",
            delta_color="normal",
        )
        m2.metric(
            "🧭 Total Jarak Nearest Neighbor (NN)",
            f"{jarak_nn:.2f} m",
            delta=f"{jarak_nn - jarak_ga:+.2f} m vs GA",
            delta_color="inverse",
        )
        m3.metric(
            "📝 Total Jarak Rute Awal (Urutan Pilih)",
            f"{jarak_awal:.2f} m",
        )
        m4.metric(
            "📍 Titik Awal (Depot)",
            f"{res['depot_label']}",
            f"({int(depot_res[0])}, {int(depot_res[1])}, Z={int(depot_res[2])})",
        )

        # Urutan Rute Kunjungan
        nama_rute_ga = df_rute_res.iloc[rute_ga]["originalLocation"].tolist()
        nama_rute_nn = df_rute_res.iloc[rute_nn]["originalLocation"].tolist()
        nama_rute_awal = df_rute_res.iloc[rute_awal]["originalLocation"].tolist()

        st.markdown("**Urutan Kunjungan Hasil Optimasi Genetic Algorithm (GA):**")
        st.code(" -> ".join(["DEPOT"] + nama_rute_ga + ["DEPOT"]), language="text")

        st.markdown("**Urutan Kunjungan Nearest Neighbor (NN):**")
        st.code(" -> ".join(["DEPOT"] + nama_rute_nn + ["DEPOT"]), language="text")

        # Grafik Konvergensi GA
        st.divider()
        st.subheader("3️⃣ Grafik Konvergensi Generasi Genetic Algorithm (GA)")
        col_chart, col_log = st.columns([2, 1])
        with col_chart:
            fig_conv = buat_grafik_konvergensi(riwayat, jarak_ga)
            st.pyplot(fig_conv, use_container_width=True)
            plt.close(fig_conv)
        with col_log:
            st.markdown("**Riwayat Jarak per 20 Generasi:**")
            st.dataframe(pd.DataFrame(log_gen), use_container_width=True, hide_index=True)

        # Visualisasi Peta Rute Gudang
        st.divider()
        st.subheader("4️⃣ Visualisasi Peta Rute Pengambilan Barang di Gudang (Z = 1)")

        tab_ga, tab_nn, tab_awal = st.tabs(
            [
                f"🗺️ Peta Rute Genetic Algorithm ({jarak_ga:.2f} m)",
                f"🧭 Peta Rute Nearest Neighbor ({jarak_nn:.2f} m)",
                f"📋 Peta Rute Awal / Urutan Pilih ({jarak_awal:.2f} m)",
            ]
        )

        with tab_ga:
            with st.spinner("Merender peta rute Genetic Algorithm..."):
                fig_ga = buat_visualisasi_rute(
                    rute_ga,
                    "Genetic Algorithm (GA)",
                    jarak_ga,
                    depot_res,
                    titik_res,
                    df_rute_res,
                    wh,
                    skenario_res,
                )
                st.pyplot(fig_ga, use_container_width=True)
                plt.close(fig_ga)

        with tab_nn:
            with st.spinner("Merender peta rute Nearest Neighbor..."):
                fig_nn = buat_visualisasi_rute(
                    rute_nn,
                    "Nearest Neighbor (NN)",
                    jarak_nn,
                    depot_res,
                    titik_res,
                    df_rute_res,
                    wh,
                    skenario_res,
                )
                st.pyplot(fig_nn, use_container_width=True)
                plt.close(fig_nn)

        with tab_awal:
            with st.spinner("Merender peta Rute Awal (Tanpa Optimasi)..."):
                fig_awal = buat_visualisasi_rute(
                    rute_awal,
                    "Rute Awal (Tanpa Optimasi)",
                    jarak_awal,
                    depot_res,
                    titik_res,
                    df_rute_res,
                    wh,
                    skenario_res,
                )
                st.pyplot(fig_awal, use_container_width=True)
                plt.close(fig_awal)


if __name__ == "__main__":
    main()
