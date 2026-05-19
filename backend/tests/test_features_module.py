from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES, to_vector


def test_feature_names_count_and_order():
    assert len(FEATURE_NAMES) == 30
    assert FEATURE_NAMES[0] == "having_IPhaving_IP_Address"
    assert FEATURE_NAMES[-1] == "Statistical_report"


def test_approximated_features_are_subset():
    assert set(APPROXIMATED_FEATURES) <= set(FEATURE_NAMES)
    assert set(APPROXIMATED_FEATURES) == {
        "web_traffic", "Page_Rank", "Google_Index",
        "Links_pointing_to_page", "Statistical_report",
    }


def test_to_vector_preserves_order():
    sample = {name: 0 for name in FEATURE_NAMES}
    sample["having_IPhaving_IP_Address"] = -1
    sample["Statistical_report"] = 1
    vec = to_vector(sample)
    assert vec[0] == -1
    assert vec[-1] == 1
    assert len(vec) == 30
