"""Qualificacao economica baseada em liquidacoes confirmadas."""

from src.domain.risk.broker_ev_evidence import BrokerEvEvidence


def test_pilot_and_promotion_are_version_sample_bound():
    empty = BrokerEvEvidence(0, ())
    assert empty.pilot_available
    assert not empty.promoted
    assert empty.lower_95 is None
    assert empty.posterior_win_probability is None
    pending = BrokerEvEvidence(200, ())
    assert not pending.pilot_available
    assert not pending.promoted
    positive = BrokerEvEvidence(200, (0.2,) * 200)
    assert positive.settled == 200
    assert positive.mean_return == 0.2
    assert positive.lower_95 == 0.2
    assert positive.promoted
    assert positive.posterior_win_probability == (200.5 / 201.0)
    insufficient = BrokerEvEvidence(199, (0.2,) * 199)
    assert not insufficient.promoted
    losing = BrokerEvEvidence(200, (-0.2,) * 200)
    assert not losing.promoted
    exploratory = BrokerEvEvidence(0, (), exploratory_purchased=200)
    assert exploratory.pilot_available
    assert not empty.suspended
    assert BrokerEvEvidence(200, (-0.2,) * 200).suspended
    assert not positive.suspended


def test_nonfinite_evidence_does_not_promote():
    evidence = BrokerEvEvidence(200, (0.2,) * 199 + (float("nan"),))
    assert evidence.lower_95 is None
    assert not evidence.promoted
