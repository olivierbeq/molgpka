from pick_a_pka import draw_pka, plot_microspecies_distribution

SMILES = "CC(=O)O"


def test_draw_pka_svg():
    svg = draw_pka(SMILES, vector=True)

    assert isinstance(svg, str)
    assert "<svg" in svg


def test_plot_distribution_svg():
    svg = plot_microspecies_distribution(SMILES, vector=True)

    assert isinstance(svg, str)
    assert "<svg" in svg
