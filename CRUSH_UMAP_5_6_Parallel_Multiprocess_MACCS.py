# -*- coding: utf-8 -*-
"""
Parallel streaming molecular fragmentation using RECAP, BRICS and CRUSH SMIRKS.
Computes MACCS keys (166-bit) fingerprints, packs them as bytes and appends to
a binary file. Metadata is streamed to CSV in batches.

FIXES applied vs original:
  1. FP_NBYTES = math.ceil(166/8) = 21  (NOT 166//8=20 — packbits pads to full bytes)
  2. Single __main__ block so PCA+UMAP is reachable.
  3. SMILES validation (NaN / empty) before submitting to workers.
  4. read_fingerprints_in_chunks: safe reshape ignoring trailing incomplete rows.
  5. Atomic write: fingerprint bytes and metadata row are written together only when
     both are valid, preventing count mismatches.
  6. PCA component count auto-clamped to min(n_components, n_samples, n_features).
  7. UMAP gracefully skipped if fewer than 2 samples.
  8. Merge uses positional alignment after explicit length check/truncation.
"""

import os
import math
import random
import time
import traceback
import multiprocessing as mp

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from sklearn.decomposition import IncrementalPCA
import umap

# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


# --------------------------
# CONFIGURATION
# --------------------------

FP_NBITS  = 166                       # MACCS keys length
FP_NBYTES = math.ceil(FP_NBITS / 8)  # = 21  ← FIX: packbits rounds UP, not floor

BATCH_WRITE = 1000
CHUNKSIZE   = 1000
N_WORKERS   = max(mp.cpu_count() - 1, 1)
MODES       = ("RECAP", "BRICS", "CRUSH")

OUT_CSV = "fragments.csv"
OUT_BIN = "fps.bin"

BATCH_SIZE        = 5000
N_COMPONENTS_PCA  = 50
N_COMPONENTS_UMAP = 2
RANDOM_SEED       = 42

FPS_BIN       = "fps.bin"
META_CSV      = "fragments.csv"
OUT_PCA       = "pca_fragments.npy"
OUT_UMAP      = "umap_fragments.npy"
OUT_CSV_EMBED = "fragments_with_umap.csv"


# ------------------------------
# Fragmentation rules (SMIRKS)
# ------------------------------

def frag_rules(mode="ALL", random_seed=None):
    recap_smirks = [
        ('Urea',                              '[#7;+0;D2,D3:1]!@C(!@=O)!@[#7;+0;D2,D3:2]>>[1*][#7:1].[2*][#7:2]'),
        ('Amide',                             '[C;!$(C([#7])[#7]):1](=!@[O:2])!@[#7;+0;!D1:3]>>[3*][C:1]=[O:2].[4*][#7:3]'),
        ('Ester',                             '[C:1](=!@[O:2])!@[O;+0:3]>>[5*][C:1]=[O:2].[6*][O:3]'),
        ('Amines',                            '[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*:1])-!@[*:2]>>[7*][*:1].[8*][*:2]'),
        ('Cyclic amines',                     '[#7;R;D3;+0:1]-!@[*:2]>>[9*][#7:1].[10*][*:2]'),
        ('Ether',                             '[#6:1]-!@[O;+0]-!@[#6:2]>>[11*][#6:1].[12*][#6:2]'),
        ('Olefin',                            '[C:1]=!@[C:2]>>[13*][C:1].[14*][C:2]'),
        ('Aromatic nitrogen - aliphatic carbon', '[n;+0:1]-!@[C:2]>>[15*][n:1].[16*][C:2]'),
        ('Lactam nitrogen - aliphatic carbon', '[O:3]=[C:4]-@[N;+0:1]-!@[C:2]>>[17*][O:3]=[C:4]-[N:1].[18*][C:2]'),
        ('Aromatic carbon - aromatic carbon', '[c:1]-!@[c:2]>>[19*][c:1].[20*][c:2]'),
        ('Aromatic nitrogen - aromatic carbon','[n;+0:1]-!@[c:2]>>[21*][n:1].[22*][c:2]'),
        ('Sulphonamide',                      '[#7;+0;D2,D3:1]-!@[S:2](=[O:3])=[O:4]>>[23*][#7:1].[24*][S:2](=[O:3])=[O:4]'),
    ]

    brics_smirks = [
        ('L1',   '[C;D3:1](=O)-;!@[#0,#6,#7,#8:2]>>[25*][*:1].[26*][*:2]'),
        ('L2a',  '[N;D3;R;$(N(@[C;!$(C=*)])@[C;!$(C=*)]):1]-[*:2]>>[27*][*:1].[28*][*:2]'),
        ('L3',   '[O;D2:1]-;!@[#0,#6,#1:2]>>[29*][*:1].[30*][*:2]'),
        ('L4',   '[C;!D1;!$(C=*):1]-;!@[#6:2]>>[31*][*:1].[32*][*:2]'),
        ('L5',   '[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O):1]-[*:2]>>[33*][*:1].[34*][*:2]'),
        ('L6',   '[C;D3;!R:1](=O)-;!@[#0,#6,#7,#8:2]>>[35*][*:1].[36*][*:2]'),
        ('L7a',  '[C;D2,D3:1]-[#6:2]>>[37*][*:1].[38*][*:2]'),
        ('L7b',  '[C;D2,D3:1]-[#6:2]>>[39*][*:1].[40*][*:2]'),
        ('L8',   '[C;!R;!D1;!$(C!-*):1]-[*:2]>>[41*][*:1].[42*][*:2]'),
        ('L9',   '[n;+0;$(n(:[c,n,o,s]):[c,n,o,s]):1]-[*:2]>>[43*][*:1].[44*][*:2]'),
        ('L10',  '[N;R;$(N(@C(=O))@[C,N,O,S]):1]-[*:2]>>[45*][*:1].[46*][*:2]'),
        ('L11',  '[S;D2:1](-;!@[#0,#6])-[*:2]>>[47*][*:1].[48*][*:2]'),
        ('L12',  '[S;D4:1]([#6,#0])(=O)(=O)-[*:2]>>[49*][*:1].[50*][*:2]'),
        ('L13',  '[C;$(C(-;@[C,N,O,S])-;@[N,O,S]):1]-[*:2]>>[51*][*:1].[52*][*:2]'),
        ('L14',  '[c;$(c(:[c,n,o,s]):[n,o,s]):1]-[*:2]>>[53*][*:1].[54*][*:2]'),
        ('L14b', '[c;$(c(:[c,n,o,s]):[n,o,s]):1]-[*:2]>>[55*][*:1].[56*][*:2]'),
        ('L15',  '[C;$(C(-;@C)-;@C):1]-[*:2]>>[57*][*:1].[58*][*:2]'),
        ('L16',  '[c;$(c(:c):c):1]-[*:2]>>[59*][*:1].[60*][*:2]'),
        ('L16b', '[c;$(c(:c):c):1]-[*:2]>>[61*][*:1].[62*][*:2]'),
    ]

    crush_smirks = [
        ('Acyclic carbonyl-N cleavage',              '[#6:1]!@[C!R:2](=[O:3])!@[N:4]>>[*:1][*:2](=[*:3]).[*:4]'),
        ('Ester',                                    '[#6:1]!@[C!R:2](=[O:3])!@[O!R:4][#6:5]>>[*:1][*:2](=[*:3]).[*:4][*:5]'),
        ('Tertiary/quaternary aliphatic amine C-N',  '[NX3,N+X4:1]([CX4:2])([CX4:3])[CX4:4]>>[*:1]([*:2])([*:3]).[*:4]'),
        ('Ring tertiary amine substituent',          '[N;D3;R:1]!@[C;!$(C=*):2]>>[*:1].[*:2]'),
        ('Urea',                                     '[N:1]!@[C!R:2](=[O:3])!@[N:4]>>[*:1][*:2](=[*:3]).[*:4]'),
        ('Ether',                                    '[#6,#1:1][#6:2]!@[O!R:3]!@[#6:4][#6,#1:5]>>[*:1][*:2][*:3].[*:4][*:5]'),
        ('Hetero-substituted alkyl C-X',             '[C;$(C(-;@[C,N,O,S])-;@[N,O,S]):1]-[*:2]>>[*:1].[*:2]'),
        ('Olefin',                                   '[C:1]!@=[C:2]>>[*:1].[*:2]'),
        ('Quaternary nitrogen',                      '[N+X4&H0:1]!@[C:2]>>[*:1].[*:2]'),
        ('Aromatic nitrogen - aliphatic carbon',     '[n,n+:1]!@[C:2]>>[*:1].[*:2]'),
        ('Lactam nitrogen - aliphatic carbon',       '[NR:1](@[CR;!$(C=[O,N]):2])(!@[C:3])@[CR:4]=[O:5]>>[*:1]([*:2])[*:4]=[*:5].[*:3]'),
        ('Aromatic carbon - aromatic carbon',        '[c:1]!@[c:2]>>[*:1].[*:2]'),
        ('Sulphonamide',                             '[S!R:1](=[O:2])(=[O:3])!@[N:4]!@[#6;!$(C=[O,N]):5]>>[*:1](=[*:2])(=[*:3]).[*:4][*:5]'),
        ('Aromatic carbon - aliphatic carbon',       '[c:1]!@[CX4;!$(C~[!#1&!#6]):2]>>[*:1].[*:2]'),
        ('Aromatic carbon - aliphatic amine',        '[c:1]!@[N:2]>>[*:1].[*:2]'),
        ('Alkyne',                                   '[C:1]!@#[C:2]>>[*:1].[*:2]'),
        ('Thioether',                                '[#6;!R:1]!@[S;X2:2]!@[#6;!R:3]>>[*:1].[*:2][*:3]'),
        ('Carbamate_1',                              '[O:1]!@[C!R:2](=[O:3])!@[N:4]>>[*:1][*:2](=[*:3]).[*:4]'),
        ('Carbamate_2',                              '[O;!R:1]!@[C;!R:2](=[O:3])!@[N;!R;!$([N]-[C](=O)-[N]):4]>>[*:1].[*:2](=[*:3]).[*:4]'),
        ('Acyl-sulphamide',                          '[#6:1][S!R:2](=[O:3])(=[O:4])!@[N:5]!@[C!R:6](=[O:7])>>[*:1][*:2](=[*:3])(=[*:4])[*:5].[*:6](=[*:7])'),
        ('Sulfacylation-O',                          '[S!R:1](=[O:2])(=[O:3])!@[O:4]!@[c:5]>>[*:1](=[*:2])(=[*:3]).[*:4][*:5]'),
        ('Aryl-C-N cleavage',                        '[c:1]!@[n:2]>>[*:1].[*:2]'),
        ('Inter-Aryl-N-N cleavage',                  '[n:1]!@[n:2]>>[*:1].[*:2]'),
        ('Ar-alkyl cleavage',                        '[c:1]-[C;!R:2]>>[*:1].[*:2]'),
        ('C-C break (aliphatic)',                    '[C;!R;!$(C=[O,N]):1]!@[C;!R;!$(C=[O,N]):2]>>[*:1].[*:2]'),
        ('C-hetero (aliphatic)',                     '[C;!R:1]!@[!#6;!R:2]>>[*:1].[*:2]'),
        ('Sulfonate ester cleavage',                 '[S!R:1](=[O:2])(=[O:3])!@[O:4]!@[C:5]>>[*:1](=[*:2])(=[*:3])[*:4].[*:5]'),
        ('Disulfide bond cleavage',                  '[S:1]-[S:2]>>[*:1].[*:2]'),
        ('Ar-vinyl-core cleavage',                   '[c:1]-[C:2]=[C:3]>>[*:1].[*:2]=[*:3]'),
        ('Ar-O cleavage',                            '[c:1]-[O:2]>>[*:1].[*:2]'),
        ('Ar-N exocyclic cleavage',                  '[c:1]-[N:2]>>[*:1].[*:2]'),
        ('Ar-S exocyclic cleavage',                  '[c:1]-[S:2]>>[*:1].[*:2]'),
        ('Ar-sp-sp C cleavage',                      '[c:1]-[C:2]#[C:3]>>[*:1].[*:2]#[*:3]'),
    ]

    rules_map = {"RECAP": recap_smirks, "BRICS": brics_smirks, "CRUSH": crush_smirks}
    if mode not in rules_map:
        raise ValueError(f"Mode must be one of: {list(rules_map.keys())}")

    rxn_smirks = rules_map[mode]
    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(rxn_smirks)
    return rxn_smirks


# --------------------------
# Helper utilities
# --------------------------

def has_smaller_number(numbers, min_frag_size):
    return any(num < min_frag_size for num in numbers)

def tryCalcNumHeavyAtoms(mol):
    try:
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    except Exception:
        return 0

def _atom_ring_count(mol):
    ring_info = mol.GetRingInfo().AtomRings()
    counts = {i: 0 for i in range(mol.GetNumAtoms())}
    for ring in ring_info:
        for idx in ring:
            counts[idx] += 1
    return counts


def read_fingerprints_in_chunks(filename, fp_bytes, batch_size):
    """
    Generator yielding batches of unpacked fingerprints as float32 numpy arrays.
    FIX: uses integer division to discard trailing incomplete rows safely.
    """
    with open(filename, "rb") as f:
        while True:
            data = f.read(fp_bytes * batch_size)
            if not data:
                break
            # FIX: discard any trailing incomplete fingerprint bytes
            n_fps = len(data) // fp_bytes
            if n_fps == 0:
                break
            arr = np.frombuffer(data[:n_fps * fp_bytes], dtype=np.uint8).reshape(n_fps, fp_bytes)
            fps = np.unpackbits(arr, axis=1)[:, :FP_NBITS]  # trim padding bits
            yield fps.astype(np.float32)


# ------------------------------
# Core recursive fragmentation
# ------------------------------

def compile_reactions_for_mode(mode, random_seed=None):
    rxn_smarts = frag_rules(mode=mode, random_seed=random_seed)
    reactions = []
    for name, smarts in rxn_smarts:
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is not None:
                reactions.append((name, rxn))
        except Exception:
            continue
    return reactions


def mol_fragmentation_single(smi, compiled_reactions, min_frag_size=3,
                              protect_neighbor_of_fused=False, conservative=True, debug=False):
    max_frag_count = 0
    best_path = []

    def recursive_fragment(smis, path, reactions):
        nonlocal max_frag_count, best_path
        fragmented = False

        for smi_local in smis:
            mol = Chem.MolFromSmiles(smi_local)
            if mol is None:
                continue

            ring_counts = _atom_ring_count(mol)
            fused_atoms = {idx for idx, cnt in ring_counts.items() if cnt >= 2}
            neighbor_of_fused = set()
            if protect_neighbor_of_fused and fused_atoms:
                for idx in fused_atoms:
                    atom = mol.GetAtomWithIdx(idx)
                    for nb in atom.GetNeighbors():
                        neighbor_of_fused.add(nb.GetIdx())
            protected_atoms = fused_atoms.union(neighbor_of_fused)

            for rxn_name, rxn in reactions:
                try:
                    if rxn is None or rxn.GetNumReactantTemplates() != 1:
                        continue
                    template = rxn.GetReactantTemplate(0)
                    matches = mol.GetSubstructMatches(template, useChirality=False)
                    if not matches:
                        continue

                    safe_matches = [m for m in matches if not any(idx in protected_atoms for idx in m)]
                    if conservative:
                        if any(any(idx in protected_atoms for idx in m) for m in matches):
                            continue
                    else:
                        if not safe_matches:
                            continue

                    rxn_products = rxn.RunReactants((mol,))
                    for products in rxn_products:
                        products_smiles = []
                        for prod in products:
                            try:
                                s = Chem.MolToSmiles(prod, True)
                                if s:
                                    products_smiles.append(s)
                            except Exception:
                                continue

                        if not products_smiles:
                            continue

                        ha_frag = [tryCalcNumHeavyAtoms(Chem.MolFromSmiles(p)) for p in products_smiles]
                        if has_smaller_number(ha_frag, min_frag_size):
                            continue

                        new_path = path + [(smi_local, rxn_name, products_smiles)]
                        total_frags = sum(len(fl) for _, _, fl in new_path)

                        if total_frags < max_frag_count:
                            continue

                        deeper_fragmented = recursive_fragment(products_smiles, new_path, reactions)
                        fragmented = True

                        if (not deeper_fragmented) and (total_frags > max_frag_count):
                            max_frag_count = total_frags
                            best_path = new_path.copy()

                except Exception:
                    continue

        return fragmented

    recursive_fragment([smi], [], compiled_reactions)
    return best_path


# --------------------------
# Fingerprint computation
# --------------------------

def compute_packed_maccs_bytes(smiles):
    """
    Compute MACCS keys (166 bits) and pack to exactly FP_NBYTES bytes.
    FIX: uses math.ceil so packed length == FP_NBYTES always.
    Returns bytes or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((FP_NBITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        packed = np.packbits(arr)          # length = ceil(166/8) = 21 bytes ✓
        assert len(packed) == FP_NBYTES, f"packed fp length mismatch: {len(packed)} != {FP_NBYTES}"
        return packed.tobytes()
    except Exception:
        return None


# --------------------------
# Worker function
# --------------------------

def worker_fragment_task(args):
    """
    Returns list of (metadata_dict, fp_bytes) pairs — keeping them together
    so counts NEVER diverge even if errors occur mid-molecule.
    """
    identifier, smi, modes_to_reactions, min_frag_size, protect_neighbor_of_fused, conservative = args

    # FIX: validate SMILES upfront
    if not isinstance(smi, str) or not smi.strip():
        return []
    mol_check = Chem.MolFromSmiles(smi)
    if mol_check is None:
        return []

    results = []  # list of (meta_dict, fp_bytes)
    try:
        for mode, reactions in modes_to_reactions.items():
            best_path = mol_fragmentation_single(
                smi, reactions,
                min_frag_size=min_frag_size,
                protect_neighbor_of_fused=protect_neighbor_of_fused,
                conservative=conservative,
                debug=False
            )
            for parent_smi, rxn_name, frag_list in best_path:
                for frag in frag_list:
                    fp_bytes = compute_packed_maccs_bytes(frag)
                    if fp_bytes is None:
                        continue
                    meta = {
                        "Identifier":       identifier,
                        "Original_SMILES":  smi,
                        "Fragment_SMILES":  frag,
                        "Rule":             rxn_name,
                        "Mode":             mode,
                    }
                    results.append((meta, fp_bytes))
    except Exception as e:
        print(f"[worker error] id={identifier}, smi={smi[:50]}, err={e}")
        traceback.print_exc()

    return results


# --------------------------
# Orchestrator
# --------------------------

def fragment_and_stream_parallel(csv_path, sep, smiles_col, id_col,
                                 out_csv=OUT_CSV, fps_bin=OUT_BIN,
                                 batch_write=BATCH_WRITE, chunksize=CHUNKSIZE,
                                 n_workers=N_WORKERS, min_frag_size=3,
                                 protect_neighbor_of_fused=True, conservative=True,
                                 random_seed=42):

    modes_to_reactions = {
        mode: compile_reactions_for_mode(mode, random_seed=random_seed)
        for mode in MODES
    }

    # clean outputs
    for path in (fps_bin, out_csv):
        if os.path.exists(path):
            os.remove(path)
    first_write = True

    total_written = 0
    rows_buffer   = []
    start         = time.time()

    pool = mp.Pool(processes=n_workers)

    with open(fps_bin, "ab") as fbin:
        reader = pd.read_csv(csv_path, sep=sep, usecols=[id_col, smiles_col],
                             chunksize=chunksize, dtype=str)
        for chunk in reader:
            tasks = [
                (row[id_col], row[smiles_col],
                 modes_to_reactions, min_frag_size,
                 protect_neighbor_of_fused, conservative)
                for _, row in chunk.iterrows()
            ]

            # FIX: worker now returns paired (meta, fp) list — no divergence possible
            for result_pairs in pool.imap_unordered(worker_fragment_task, tasks):
                for meta, fp_bytes in result_pairs:
                    fbin.write(fp_bytes)          # write FP
                    rows_buffer.append(meta)       # write meta in sync
                    total_written += 1

                if len(rows_buffer) >= batch_write:
                    pd.DataFrame(rows_buffer).to_csv(
                        out_csv, mode="a", index=False, header=first_write)
                    first_write = False
                    rows_buffer = []

                if total_written and total_written % (batch_write * 10) < len(result_pairs) + 1:
                    elapsed = (time.time() - start) / 60.0
                    print(f"[{time.strftime('%H:%M:%S')}] fragments written: {total_written:,}  "
                          f"elapsed: {elapsed:.2f} min")

        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(
                out_csv, mode="a", index=False, header=first_write)

    pool.close()
    pool.join()

    elapsed_total = (time.time() - start) / 60.0
    print(f"\nFragmentation done. Metadata → '{out_csv}', FPs → '{fps_bin}'.")
    print(f"Total fragments: {total_written:,}  |  Elapsed: {elapsed_total:.2f} min")
    return total_written


# --------------------------
# Incremental PCA + UMAP
# --------------------------

def run_incremental_pca_and_umap_robust(
    fps_bin_path=FPS_BIN,
    fp_bytes=FP_NBYTES,
    batch_size=BATCH_SIZE,
    n_components_pca=N_COMPONENTS_PCA,
    n_components_umap=N_COMPONENTS_UMAP,
    random_seed=RANDOM_SEED,
    out_pca=OUT_PCA,
    out_umap=OUT_UMAP,
    meta_csv=META_CSV,
    out_csv_embed=OUT_CSV_EMBED,
    umap_n_neighbors=15,
    umap_min_dist=0.1
):
    print("\n=== Running Incremental PCA + UMAP ===\n")

    # --- Step 1: dataset size ---
    if not os.path.exists(fps_bin_path):
        raise FileNotFoundError(f"Binary file not found: {fps_bin_path}")

    total_bytes = os.path.getsize(fps_bin_path)
    total_fps   = total_bytes // fp_bytes
    print(f"[INFO] Fingerprints detected: {total_fps:,}  ({total_bytes / 1e6:.2f} MB)")

    if total_fps == 0:
        raise RuntimeError("No fingerprints found in binary file. Aborting.")

    # FIX: clamp PCA components to valid range
    n_features        = FP_NBITS
    n_components_pca  = min(n_components_pca, total_fps - 1, n_features)
    if n_components_pca < 1:
        n_components_pca = 1
    print(f"[INFO] n_components_pca clamped to {n_components_pca}")

    batch_size = min(batch_size, total_fps)

    # --- Step 2: Fit IncrementalPCA ---
    print(f"[INFO] Fitting IncrementalPCA (n_components={n_components_pca}, batch_size={batch_size})...")
    ipca      = IncrementalPCA(n_components=n_components_pca, batch_size=batch_size)
    processed = 0

    for fps in read_fingerprints_in_chunks(fps_bin_path, fp_bytes, batch_size):
        ipca.partial_fit(fps)
        processed += fps.shape[0]
        print(f"[INFO]   PCA fit: {processed:,}/{total_fps:,}", end="\r")
    print()

    # --- Step 3: Transform ---
    print("[INFO] Transforming fingerprints to PCA space...")
    pca_chunks = []
    processed  = 0

    for fps in read_fingerprints_in_chunks(fps_bin_path, fp_bytes, batch_size):
        pca_chunks.append(ipca.transform(fps))
        processed += fps.shape[0]
        print(f"[INFO]   PCA transform: {processed:,}/{total_fps:,}", end="\r")
    print()

    X_pca = np.vstack(pca_chunks)
    np.save(out_pca, X_pca)
    print(f"[INFO] PCA saved → {out_pca}  shape={X_pca.shape}")

    # --- Step 4: UMAP ---
    # FIX: guard against degenerate datasets
    if X_pca.shape[0] < 2:
        print("[WARN] Too few samples for UMAP. Skipping.")
        X_umap = np.zeros((X_pca.shape[0], n_components_umap), dtype=np.float32)
    else:
        # clamp n_neighbors so it never exceeds dataset size
        n_neighbors_safe = min(umap_n_neighbors, X_pca.shape[0] - 1)
        print(f"[INFO] Running UMAP (n_neighbors={n_neighbors_safe}, min_dist={umap_min_dist})...")
        reducer = umap.UMAP(
            n_components=n_components_umap,
            n_neighbors=n_neighbors_safe,
            min_dist=umap_min_dist,
            metric="euclidean",
            init="random",
            random_state=random_seed,
            verbose=True,
        )
        try:
            X_umap = reducer.fit_transform(X_pca)
        except MemoryError:
            print("[ERROR] UMAP ran out of memory. Try reducing batch_size or n_components_pca.")
            raise

    np.save(out_umap, X_umap)
    print(f"[INFO] UMAP saved → {out_umap}  shape={X_umap.shape}")

    # --- Step 5: Merge with metadata ---
    print("[INFO] Merging metadata with UMAP embeddings...")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")

    meta_df = pd.read_csv(meta_csv)

    # FIX: explicit alignment check with informative warning
    if len(meta_df) != X_umap.shape[0]:
        print(f"[WARN] Row count mismatch — metadata={len(meta_df)}, embeddings={X_umap.shape[0]}. "
              f"Truncating to shorter length.")
        min_len = min(len(meta_df), X_umap.shape[0])
        meta_df = meta_df.iloc[:min_len].copy()
        X_umap  = X_umap[:min_len]

    meta_df["UMAP_1"] = X_umap[:, 0]
    meta_df["UMAP_2"] = X_umap[:, 1]
    meta_df.to_csv(out_csv_embed, index=False)
    print(f"[INFO] Combined CSV saved → {out_csv_embed}")

    print("\n✅ Incremental PCA + UMAP completed successfully.\n")
    return X_pca, X_umap


# --------------------------
# CLI entry point  (FIX: single __main__ block)
# --------------------------

if __name__ == "__main__":
    print("=== Parallel Fragmentation + Incremental PCA + UMAP ===\n")

    csv_path   = input("Enter input CSV path: ").strip()
    sep        = input("Enter CSV separator (default ','): ").strip() or ","
    smiles_col = input("Enter SMILES column name: ").strip()
    id_col     = input("Enter identifier column name: ").strip()

    n_written = fragment_and_stream_parallel(
        csv_path, sep, smiles_col, id_col,
        out_csv=OUT_CSV, fps_bin=OUT_BIN,
        batch_write=BATCH_WRITE, chunksize=CHUNKSIZE,
        n_workers=N_WORKERS, min_frag_size=3,
        protect_neighbor_of_fused=True, conservative=True,
        random_seed=42,
    )

    if n_written == 0:
        print("\n[WARN] No fragments were generated. Check your input SMILES and column names.")
        print("       Skipping PCA+UMAP step.")
    else:
        choice = input("\nRun PCA+UMAP? (y/n): ").strip().lower()
        if choice == "y":
            run_incremental_pca_and_umap_robust(
                fps_bin_path=OUT_BIN,
                fp_bytes=FP_NBYTES,
                batch_size=BATCH_SIZE,
                n_components_pca=N_COMPONENTS_PCA,
                n_components_umap=N_COMPONENTS_UMAP,
                random_seed=RANDOM_SEED,
                out_pca=OUT_PCA,
                out_umap=OUT_UMAP,
                meta_csv=OUT_CSV,
                out_csv_embed=OUT_CSV_EMBED,
            )
        else:
            print("Skipping PCA+UMAP. Fragmentation outputs are ready.")