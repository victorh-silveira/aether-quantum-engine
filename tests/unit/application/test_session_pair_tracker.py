from src.application.services.orchestrator.session_pair_tracker import SessionPairTracker


def test_session_pair_tracker_initial_empty():
    tracker = SessionPairTracker()
    stats = tracker.get_symbol_stats("OTC_SSMI")
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["streak"] == 0
    assert stats["pnl"] == 0.0
    assert tracker.symbol_rotation_adjustment("OTC_SSMI") == 0.0


def test_session_pair_tracker_record_wins():
    tracker = SessionPairTracker()
    tracker.record_symbol_settlement("OTC_SSMI", won=True, profit=8.5)
    stats = tracker.get_symbol_stats("OTC_SSMI")
    assert stats["wins"] == 1
    assert stats["streak"] == 1
    assert stats["pnl"] == 8.5
    assert tracker.symbol_rotation_adjustment("OTC_SSMI") == 0.04

    tracker.record_symbol_settlement("OTC_SSMI", won=True, profit=8.5)
    assert tracker.get_symbol_stats("OTC_SSMI")["streak"] == 2
    assert tracker.symbol_rotation_adjustment("OTC_SSMI") == 0.08

    tracker.record_symbol_settlement("OTC_SSMI", won=True, profit=8.5)
    tracker.record_symbol_settlement("OTC_SSMI", won=True, profit=8.5)
    assert tracker.symbol_rotation_adjustment("OTC_SSMI") == 0.12


def test_session_pair_tracker_record_losses():
    tracker = SessionPairTracker()
    tracker.record_symbol_settlement("OTC_SPC", won=False, profit=-10.0)
    stats = tracker.get_symbol_stats("OTC_SPC")
    assert stats["losses"] == 1
    assert stats["streak"] == -1
    assert stats["pnl"] == -10.0
    assert tracker.symbol_rotation_adjustment("OTC_SPC") == -0.10

    tracker.record_symbol_settlement("OTC_SPC", won=False, profit=-10.0)
    assert tracker.symbol_rotation_adjustment("OTC_SPC") == -0.20

    tracker.record_symbol_settlement("OTC_SPC", won=False, profit=-10.0)
    assert tracker.symbol_rotation_adjustment("OTC_SPC") == -0.25


def test_session_pair_tracker_turnaround_and_reset():
    tracker = SessionPairTracker()
    tracker.record_symbol_settlement("OTC_NDX", won=False, profit=-10.0)
    assert tracker.get_symbol_stats("OTC_NDX")["streak"] == -1

    tracker.record_symbol_settlement("OTC_NDX", won=True, profit=8.5)
    assert tracker.get_symbol_stats("OTC_NDX")["streak"] == 1

    tracker.record_symbol_settlement("OTC_NDX", won=False, profit=-10.0)
    assert tracker.get_symbol_stats("OTC_NDX")["streak"] == -1

    tracker.reset()
    assert tracker.get_symbol_stats("OTC_NDX")["wins"] == 0

    tracker._stats["OTC_SSMI"] = {"streak": 0}
    assert tracker.symbol_rotation_adjustment("OTC_SSMI") == 0.0
