import pytest
import torch


class TestGCNConvCachedPath:
    """Exercise the cached branch in GCNConv.forward."""

    def _make_graph(self, n=4):
        """Create a tiny fully-connected graph."""
        x = torch.randn(n, 8)
        # Complete graph edges (both directions)
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return x, edge_index

    def test_cached_forward_second_call_reuses_cache(self):
        from pick_a_pka.backends.molgpka.network import GCNConv
        conv = GCNConv(8, 4, cached=True)
        conv.eval()
        x, edge_index = self._make_graph()
        with torch.no_grad():
            out1 = conv(x, edge_index)
            out2 = conv(x, edge_index)  # hits the cached branch
        assert torch.allclose(out1, out2)

    def test_cached_forward_raises_on_edge_count_change(self):
        from pick_a_pka.backends.molgpka.network import GCNConv
        conv = GCNConv(8, 4, cached=True)
        conv.eval()
        x, edge_index = self._make_graph(n=4)
        with torch.no_grad():
            conv(x, edge_index)  # populate cache

        # Now pass a different number of edges — must raise RuntimeError
        x2, edge_index2 = self._make_graph(n=3)
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                conv(x2, edge_index2)

    def test_no_bias_variant(self):
        from pick_a_pka.backends.molgpka.network import GCNConv
        conv = GCNConv(8, 4, bias=False)
        assert conv.bias is None
        x, edge_index = self._make_graph()
        with torch.no_grad():
            out = conv(x, edge_index)
        assert out.shape == (4, 4)
