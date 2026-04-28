import io
import re
import math
import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import cairosvg
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

from .predict_pka import predict, calculate_microspecies_abundances


def draw_pka(mol: Chem.Mol, uncharged: bool = True, image_size=(800, 800), padding: float = 0.1, vector: bool = True) -> str | Image.Image:
    # 1. Prediction and Setup
    mol_copy = Chem.Mol(mol)

    # Assuming 'predict' is defined in your environment
    base_pka, acid_pka, mol_copy = predict(mol_copy, uncharged=uncharged)

    rdDepictor.SetPreferCoordGen(False)
    rdDepictor.Compute2DCoords(mol_copy)

    mol_prepared = rdMolDraw2D.PrepareMolForDrawing(mol_copy)
    conf = mol_prepared.GetConformer()
    N_atoms = mol_prepared.GetNumAtoms()

    if N_atoms == 0:
        return Image.new("RGB", image_size, (255, 255, 255))

    drawer = (rdMolDraw2D.MolDraw2DSVG if vector else rdMolDraw2D.MolDraw2DCairo)(image_size[0], image_size[1])
    draw_opts = drawer.drawOptions()
    draw_opts.clearBackground = False
    draw_opts.addAtomIndices = False

    # Since we are keeping labels closer, padding can be slightly reduced
    draw_opts.padding = padding

    # Draw the underlying molecule first
    drawer.DrawMolecule(mol_prepared)

    # 2. Pre-calculate atom coordinates and Center of Geometry (CoG)
    atom_positions = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(N_atoms)]
    cog_x = sum(p[0] for p in atom_positions) / N_atoms
    cog_y = sum(p[1] for p in atom_positions) / N_atoms

    # 3. Combine Dictionaries & Initialize ONE Label Group per Atom
    combined_pka = {}
    for idx, pka in acid_pka.items():
        combined_pka.setdefault(idx, []).append((pka, (0.6, 0.0, 0.3, 1.0)))  # Dark Red
    for idx, pka in base_pka.items():
        combined_pka.setdefault(idx, []).append((pka, (0.0, 0.2, 0.8, 1.0)))  # Dark Blue

    labels = []
    for atom_idx, pka_list in combined_pka.items():
        if atom_idx >= N_atoms: continue

        pos = atom_positions[atom_idx]
        atom = mol_prepared.GetAtomWithIdx(atom_idx)

        # Local vector (pointing away from immediate neighbors)
        v_local_x, v_local_y = 0.0, 0.0
        neighbors = atom.GetNeighbors()
        if neighbors:
            for n in neighbors:
                nx, ny = atom_positions[n.GetIdx()]
                dx, dy = nx - pos[0], ny - pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    v_local_x += dx / dist
                    v_local_y += dy / dist
            v_local_x, v_local_y = -v_local_x, -v_local_y
        else:
            v_local_x, v_local_y = 0.0, -1.0

        # Global vector (pointing away from the center of the molecule)
        v_glob_x, v_glob_y = pos[0] - cog_x, pos[1] - cog_y
        dist_glob = math.hypot(v_glob_x, v_glob_y)
        if dist_glob > 1e-4:
            v_glob_x /= dist_glob
            v_glob_y /= dist_glob

        # Blended Vector: Guarantees the label starts by pointing to the "outside" of the molecule
        vx, vy = v_local_x + 0.8 * v_glob_x, v_local_y + 0.8 * v_glob_y
        v_len = math.hypot(vx, vy)
        if v_len > 1e-4:
            vx /= v_len
            vy /= v_len
        else:
            vx, vy = 1.0, 0.0

        num_hs = atom.GetTotalNumHs()
        degree = atom.GetDegree()

        # Calculate a safe radius to clear the heavy atom and its implicit hydrogens
        safe_radius = 0.5 + (0.3 * num_hs)

        # Extra bump ONLY for tertiary/quaternary atoms (to clear methyl bonds)
        if degree >= 3:
            safe_radius += 0.5

        if len(pka_list) > 1:
            safe_radius += 0.35 * (len(pka_list) - 1)

        # Horizontal width clearance
        safe_radius += 0.15 * abs(vx)

        # The origin MUST remain the exact atom center so the line points correctly!
        origin_x, origin_y = pos[0], pos[1]

        init_dist = safe_radius + 0.5
        labels.append({
            "origin": (origin_x, origin_y),
            "safe_radius": safe_radius,
            "pos": [origin_x + vx * safe_radius, origin_y + vy * safe_radius],
            "pka_list": pka_list,
            "n_lines": len(pka_list)
        }
        )

    # 4. The Force-Directed Relaxation Engine
    for _ in range(500):
        displacements = [[0.0, 0.0] for _ in labels]

        for i, lbl in enumerate(labels):
            fx, fy = 0.0, 0.0
            lx, ly = lbl["pos"]

            # A. Repel from Atoms (Reduced radius so they can sit closer)
            for ax, ay in atom_positions:
                # Do not repel from the parent atom!
                # The spring completely handles the parent distance.
                if (ax, ay) == lbl["origin"]:
                    continue
                dx, dy = lx - ax, ly - ay
                if dx == 0 and dy == 0: dx, dy = 0.01, 0.01
                # Divide dx by 1.5 to push harder horizontally, protecting the wide text
                d = math.hypot(dx / 1.5, dy)
                if d < 0.8:
                    force = (0.8 - d) * 0.5
                    fx += (dx / d) * force
                    fy += (dy / d) * force

            # B. Repel from other Label Groups
            for j, o_lbl in enumerate(labels):
                if i == j: continue
                ox, oy = o_lbl["pos"]
                dx, dy = lx - ox, ly - oy
                if dx == 0 and dy == 0: dx, dy = random.random() * 0.1, random.random() * 0.1

                # If a group has many lines, it's taller, so repel harder vertically
                avg_lines = (lbl["n_lines"] + o_lbl["n_lines"]) / 2.0
                dy_adj = dy / (0.8 + 0.2 * avg_lines)
                d = math.hypot(dx / 1.5, dy_adj)
                if d < 1.4:
                    force = (1.4 - d) * 1.5
                    fx += (dx / d) * force
                    fy += (dy_adj / d) * force

            # C. Anchor Attractor (Spring to keep them near their origin atom)
            orig_x, orig_y = lbl["origin"]
            dx, dy = orig_x - lx, orig_y - ly
            d = math.hypot(dx, dy)
            # Pull exactly to the safe radius
            target_distance = lbl["safe_radius"]
            if d > 1e-4:
                force = (d - target_distance) * 0.6  # Stronger spring (0.4)
                fx += (dx / d) * force
                fy += (dy / d) * force

            displacements[i] = [fx, fy]

        # Apply the physics movements
        for i, lbl in enumerate(labels):
            lbl["pos"][0] += displacements[i][0] * 0.3
            lbl["pos"][1] += displacements[i][1] * 0.3

    # 5. Draw the Finalized Lines and Stacked Text
    for lbl in labels:
        lx, ly = lbl["pos"]
        orig_x, orig_y = lbl["origin"]
        safe_rad = lbl["safe_radius"]

        dx, dy = lx - orig_x, ly - orig_y
        d = math.hypot(dx, dy)

        # Draw the callout line ONLY if the text is further than the safe radius
        if d > safe_rad + 0.15:
            # Start the line at the edge of the safe radius so it doesn't strike through Hs
            start_pt = Point2D(orig_x + (dx / d) * (safe_rad - 0.1), orig_y + (dy / d) * (safe_rad - 0.1))
            # End the line slightly before the center of the text
            end_pt = Point2D(lx - (dx / d) * 0.35, ly - (dy / d) * 0.35)
            drawer.SetColour((0.6, 0.6, 0.6, 1.0))
            drawer.DrawLine(start_pt, end_pt)

        # Draw the Stacked Text
        pka_list = lbl["pka_list"]
        n_lines = len(pka_list)

        # Adjust this if the vertical gap between lines is too big/small
        line_height = 0.65

        # Calculate starting Y to ensure the whole text block is centered vertically on the callout line
        start_y = ly - (n_lines - 1) * line_height / 2.0

        for i, (pka_val, color) in enumerate(pka_list):
            # RDKit's DrawString automatically center-aligns horizontally around the provided X coordinate
            text_pt = Point2D(lx, start_y + i * line_height)
            drawer.SetColour(color)
            drawer.DrawString(f"pKa: {pka_val:.1f}", text_pt)

    # 6. Finalize and convert to Image
    drawer.FinishDrawing()
    content = drawer.GetDrawingText()
    if vector:
        return content
    return Image.open(io.BytesIO(content))


def plot_microspecies_distribution(mol: Chem.Mol, vector: bool = True) -> str | Image.Image:
    """Plot the fractional abundance of different ionization states of a molecule
    as a function of pH, based on its pKa values.

    :param vector: should the value returned be raw SVG data?
    :return: The raw SVG string if `vector` is True`, else a PIL Image object.
    """
    # Calculate relative abundances
    abundance_data: dict[float, dict[float, Chem.Mol]] = calculate_microspecies_abundances(
        mol, ph_range=(0, 14), ph_step=0.05
    )
    base_pka_dict, acid_pka_dict, _ = predict(mol, uncharged=True)

    # Extract coordinates for matplotlib
    X_pH = sorted(abundance_data.keys())

    # Extract curves
    num_states = len(abundance_data[X_pH[0]])
    Y_abundances = []
    for state_idx in range(num_states):
        # Extract the abundance values for this specific state across all pHs
        y_curve = [list(abundance_data[ph].keys())[state_idx] for ph in X_pH]
        Y_abundances.append(y_curve)

    # Colors matching standard chemistry plotting tools (Red, Blue, Orange, Green, Purple...)
    #         Vibrant colours first
    #         #alizarin  #curious blue  #solid orange  #mint green  #amethyst  #prussian blue  #Moroccan ruby  #raspberry yogurt pink  #aged moustache grey  #olive     #cyan      #bright raspberry   #purple
    colors = ['#e74c3c', '#3498db',     '#f39c12',     '#2ecc71',   '#9b59b6', '#34495e',      '#8c564b',      '#e377c2',              '#7e7e7e',            '#bcbd22', '#17becf', '#b8105a',          '#620f77',
              # Brighter colours
              '#ff9896',  '#aec7e8',     '#ffbb78',     '#98df8a',   '#c5b0d5', '#56799c',      '#c49c94',      '#f7b6d2',              '#c7c7c7',            '#dbdb8d', '#9edae4', '#ff4f9b',          '#cf99ff']

    linestyles = ['-', '--', '-.', ':']

    # Dynamic Layout Math (Calculates rows needed and exact inches)
    max_cols = 6
    num_rows = math.ceil(num_states / max_cols)

    fig_width_in = 14
    plot_height_in = 5.0  # The graph will always be 5 inches tall
    row_height_in = 2.4  # Each row of molecules takes 2.4 inches
    text_space_in = 1.0  # 1 inch at the bottom for the pKa text
    top_margin_in = 0.5

    bottom_margin_in = (num_rows * row_height_in) + text_space_in
    fig_height_in = plot_height_in + bottom_margin_in + top_margin_in

    # Convert layout inches to Matplotlib fractions
    plot_left_margin_frac = 0.4
    plot_width_frac = 0.58

    ax_height_frac = plot_height_in / fig_height_in
    bottom_margin_frac = bottom_margin_in / fig_height_in

    # Create the Plot
    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    ax = fig.add_axes([plot_left_margin_frac, bottom_margin_frac, plot_width_frac, ax_height_frac])

    # Plot the curves
    for i in range(num_states):
        c_idx = i % len(colors)
        ls_idx = (i // len(colors)) % len(linestyles)
        ax.plot(X_pH, Y_abundances[i],
                lw=2.5,
                color=colors[c_idx],
                linestyle=linestyles[ls_idx])


    # Styling to match the provided image
    ax.set_xlim(0, 14.01)
    ax.set_ylim(0, 101)
    ax.set_xlabel('pH', fontsize=10, color='#333333')
    ax.set_ylabel('Microspecies distribution (%)', fontsize=10, color='#333333')

    # Ticks every 2 for X, every 10 for Y
    ax.set_xticks(np.arange(0, 15, 2))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis='both', colors='#555555')

    # Grid styling
    ax.grid(True, linestyle='-', alpha=0.4, color='#cccccc')

    # Remove top and right borders for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')

    # Add textual information to the plot (Strongest acid/base pKa)
    if acid_pka_dict:
        min_acid = min(acid_pka_dict.values())
        fig.text(0.085, 0.6 / fig_height_in, f"Strongest acidic pKa: {min_acid:.2f}", fontsize=10, ha='left')
    if base_pka_dict:
        max_base = max(base_pka_dict.values())
        fig.text(0.085, 0.2 / fig_height_in, f"Strongest basic pKa: {max_base:.2f}", fontsize=10, ha='left')

    # Space the molecules evenly across the bottom of the graph
    grid_x = np.linspace(1, 13, max_cols)

    # 4. Generate the pure Matplotlib SVG String
    buf = io.StringIO()
    fig.savefig(buf, format='svg', transparent=True)
    plt.close(fig)
    mpl_svg = buf.getvalue()

    # Embed microspecies molecules
    state_molecules = list(abundance_data[X_pH[0]].values())
    injections = []

    # Target size for the molecules in SVG points
    mol_size_pt = 160
    internal_res = 1000  # MASSIVE internal resolution for ultra-fine lines

    # Margins for the microspecies span the full canvas width
    micro_left_margin_frac = 0.08
    micro_width_frac = 0.88

    for i, state_mol in enumerate(state_molecules):
        c_idx = i % len(colors)
        ls_idx = (i // len(colors)) % len(linestyles)

        # Determine Grid Position
        row = i // max_cols
        col = i % max_cols
        x_val = grid_x[col]

        # Render the 2D molecule as a Vector SVG without padding
        drawer = rdMolDraw2D.MolDraw2DSVG(internal_res, internal_res)
        opts = drawer.drawOptions()
        opts.padding = 0.1  # Removes the huge white box boundary!
        opts.clearBackground = False  # Transparent background
        opts.bondLineWidth = 3  # Force lines to be thin, which scales down well

        if hasattr(opts, 'maxFontSize'):
            opts.maxFontSize = 24  # Keep labels readable at high res

        mol_prep = rdMolDraw2D.PrepareMolForDrawing(state_mol)
        drawer.DrawMolecule(mol_prep)
        drawer.FinishDrawing()

        svg_mol = drawer.GetDrawingText()

        # Isolate the core <svg> payload and make it dynamically resizable
        start_idx = svg_mol.find('<svg')
        svg_mol = svg_mol[start_idx:]
        svg_mol = re.sub(r"width='.*?px'", "width='100%'", svg_mol, count=1)
        svg_mol = re.sub(r"height='.*?px'", "height='100%'", svg_mol, count=1)

        # Calculate exact absolute pixel (pt) coordinates for injection
        # X mapping
        frac_x = micro_left_margin_frac + (x_val / 14) * micro_width_frac
        x_pt = frac_x * fig_width_in * 72.0  # 72 points per inch in SVG

        # Y mapping (origin is TOP in SVG)
        y_center_in = top_margin_in + plot_height_in + (row + 0.4) * row_height_in
        y_pt = y_center_in * 72.0

        # Inject the Molecule SVG
        anchor_x = x_pt - (mol_size_pt / 2)
        anchor_y = y_pt - (mol_size_pt / 2)

        injection = f'<svg x="{anchor_x}" y="{anchor_y}" width="{mol_size_pt}" height="{mol_size_pt}">\n{svg_mol}\n</svg>'
        injections.append(injection)

        # Inject the colored underline into the SVG as a basic line segment
        line_y_pt = (top_margin_in + plot_height_in + (row + 0.88) * row_height_in) * 72.0
        line_w_pt = 60  # Width of the underline

        dash_str = ""
        if linestyles[ls_idx] == '--':
            dash_str = 'stroke-dasharray="8,4"'
        elif linestyles[ls_idx] == '-.':
            dash_str = 'stroke-dasharray="8,4,2,4"'
        elif linestyles[ls_idx] == ':':
            dash_str = 'stroke-dasharray="3,3"'

        line_svg = f'<line x1="{x_pt - line_w_pt}" y1="{line_y_pt}" x2="{x_pt + line_w_pt}" y2="{line_y_pt}" ' \
                   f'stroke="{colors[c_idx]}" stroke-width="4" {dash_str} />'
        injections.append(line_svg)

    # Embed pKa-annotated molecule on the left
    annotated_svg = draw_pka(mol, vector=True, image_size=(internal_res, internal_res), padding=0.075)
    # Clean the XML headers
    start_idx = annotated_svg.find('<svg')
    if start_idx != -1:
        annotated_svg = annotated_svg[start_idx:]
    # Make it responsive
    annotated_svg = re.sub(r"width='.*?px'", "width='100%'", annotated_svg, count=1)
    annotated_svg = re.sub(r"height='.*?px'", "height='100%'", annotated_svg, count=1)

    # 1. Define exactly how much empty space (in points) you want between the molecule and the plot
    margin_gap_pt = 35.0
    left_padding_pt = 5.0  # Small buffer from the absolute left edge of the image

    # Scale to 98% of the allocated left area width
    annotated_area_width_in = fig_width_in * plot_left_margin_frac
    annotated_size_pt = annotated_area_width_in * 72.0 * 0.98 - margin_gap_pt - left_padding_pt

    # Allow the molecule to expand significantly more vertically before capping
    max_allowed_height_pt = plot_height_in * 72.0 * 1.3
    if annotated_size_pt > max_allowed_height_pt:
        annotated_size_pt = max_allowed_height_pt

    # Center horizontally in the left margin
    annotated_x_pt = left_padding_pt # (annotated_area_width_in * 72.0 - annotated_size_pt) / 2

    # Center vertically relative to the plot
    plot_center_y_pt = (top_margin_in + plot_height_in / 2.0) * 72.0
    annotated_y_pt = plot_center_y_pt - (annotated_size_pt / 2.0)

    injections.append(f'<svg x="{annotated_x_pt}" y="{annotated_y_pt}" '
                      f'width="{annotated_size_pt}" height="{annotated_size_pt}">\n{annotated_svg}\n</svg>'
                      )

    # Finalize the composite SVG
    end_tag_idx = mpl_svg.rfind('</svg>')
    final_svg = mpl_svg[:end_tag_idx] + "\n".join(injections) + "\n</svg>"

    if vector:
        return final_svg

    img = BytesIO()
    cairosvg.svg2png(bytestring=final_svg, write_to=img, dpi=300)
    return Image.open(img)
