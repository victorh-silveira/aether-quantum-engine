from src.application.services.llm.deriv_timeframes import deriv_tf_compact_numeric_tag, deriv_tf_label


def test_deriv_tf_label_mapeia_ticks_deriv_padrao():
    assert deriv_tf_label(86400) == "D1"
    assert deriv_tf_label(3600) == "H1"
    assert deriv_tf_label(60) == "M1"


def test_deriv_tf_label_nao_positivo_retorna_placeholder():
    assert deriv_tf_label(0) == "TF?"
    assert deriv_tf_label(-5) == "TF?"


def test_deriv_tf_label_subminuto_usa_prefixo_s():
    assert deriv_tf_label(30) == "s30"


def test_deriv_tf_label_minutos_inferior_a_60():
    assert deriv_tf_label(450) == "M7"


def test_deriv_tf_label_grande_sem_slot_usa_suffix_m():
    assert deriv_tf_label(5400) == "90m"


def test_deriv_tf_label_multiplo_de_60_horas_acima_de_24():
    assert deriv_tf_label(90000) == "1500m"


def test_deriv_tf_compact_numeric_tag_extremos():
    assert deriv_tf_compact_numeric_tag(0) == "?"
    assert deriv_tf_compact_numeric_tag(-3) == "?"
    assert deriv_tf_compact_numeric_tag(45) == "1"
    assert deriv_tf_compact_numeric_tag(3600) == "60"
