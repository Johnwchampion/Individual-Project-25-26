import json
import math
import numpy as np
import pytest

from candidates import select_candidates, load_rd_scores


def _write_rd(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps({k: v.tolist() for k, v in data.items()}))
    return str(p)


def _layer(i):
    return f"model.layers.{i}.mlp"


class TestSelectCandidates:

    def test_basic_intersection(self, tmp_path):
        # Expert 0 is the most negative in both metrics → should be selected
        freq  = {_layer(0): np.array([-0.5,  0.1,  0.2])}
        logit = {_layer(0): np.array([-0.4,  0.3,  0.1])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=1, direction="negative",
        )
        assert result == {0: [0]}

    def test_no_intersection(self, tmp_path):
        # Expert 0 top in freq, expert 1 top in logit → disjoint → layer absent
        freq  = {_layer(0): np.array([-0.5,  0.1,  0.2])}
        logit = {_layer(0): np.array([ 0.1, -0.5,  0.2])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=1, direction="negative",
        )
        assert result == {}

    def test_positive_direction(self, tmp_path):
        # Positive direction: highest experts selected (expert 1 highest in both)
        freq  = {_layer(0): np.array([0.1,  0.5, -0.3])}
        logit = {_layer(0): np.array([0.2,  0.4, -0.1])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=1, direction="positive",
        )
        assert result == {0: [1]}

    def test_multiple_layers(self, tmp_path):
        # Two layers, each with one clear candidate
        freq  = {_layer(0): np.array([-0.5, 0.1]), _layer(5): np.array([0.1, -0.5])}
        logit = {_layer(0): np.array([-0.4, 0.2]), _layer(5): np.array([0.2, -0.4])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=1, direction="negative",
        )
        assert 0 in result and 5 in result
        assert result[0] == [0]
        assert result[5] == [1]

    def test_layer_index_parsed_correctly(self, tmp_path):
        freq  = {_layer(12): np.array([-0.5, 0.1, 0.2])}
        logit = {_layer(12): np.array([-0.4, 0.3, 0.1])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=1, direction="negative",
        )
        assert 12 in result

    def test_n_larger_than_experts(self, tmp_path):
        # N=10 but only 3 experts; should not crash and should return full intersection
        freq  = {_layer(0): np.array([-0.5, -0.3, -0.1])}
        logit = {_layer(0): np.array([-0.4, -0.2, -0.1])}
        result = select_candidates(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
            n=10, direction="negative",
        )
        assert set(result[0]) == {0, 1, 2}


class TestLoadRdScores:

    def test_mean_is_average_of_freq_and_logit(self, tmp_path):
        freq  = {_layer(0): np.array([1.0, -1.0, 0.0])}
        logit = {_layer(0): np.array([3.0, -3.0, 0.0])}
        # Expected mean before normalisation: [2.0, -2.0, 0.0]
        # std of [2, -2, 0] = sqrt((4+4+0)/3) ≈ 1.6329...
        result = load_rd_scores(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
        )
        expected_mean = np.array([2.0, -2.0, 0.0])
        std = expected_mean.std()
        expected_normalised = expected_mean / std
        for i in range(3):
            assert math.isclose(result[0][i], expected_normalised[i], rel_tol=1e-5)

    def test_z_normalisation_unit_std(self, tmp_path):
        freq  = {_layer(0): np.array([1.0, 2.0, 3.0, 4.0])}
        logit = {_layer(0): np.array([1.0, 2.0, 3.0, 4.0])}
        result = load_rd_scores(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
        )
        values = np.array(list(result[0].values()))
        assert math.isclose(values.std(), 1.0, rel_tol=1e-5)

    def test_zero_std_does_not_crash(self, tmp_path):
        # All-zero arrays: std=0, must not raise ZeroDivisionError
        freq  = {_layer(0): np.array([0.0, 0.0, 0.0])}
        logit = {_layer(0): np.array([0.0, 0.0, 0.0])}
        result = load_rd_scores(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
        )
        assert 0 in result
        for v in result[0].values():
            assert v == 0.0

    def test_return_structure(self, tmp_path):
        freq  = {_layer(3): np.array([0.1, 0.2, 0.3])}
        logit = {_layer(3): np.array([0.4, 0.5, 0.6])}
        result = load_rd_scores(
            _write_rd(tmp_path, "f.json", freq),
            _write_rd(tmp_path, "l.json", logit),
        )
        assert isinstance(result, dict)
        assert all(isinstance(k, int) for k in result)
        assert all(isinstance(v, dict) for v in result.values())
        assert set(result[3].keys()) == {0, 1, 2}
        assert all(isinstance(v, float) for v in result[3].values())
