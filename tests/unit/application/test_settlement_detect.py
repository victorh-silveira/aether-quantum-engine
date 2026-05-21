from src.application.services.orchestrator.settlement_detect import contract_payload_is_settled


def test_contract_payload_is_settled_flags():
    assert contract_payload_is_settled({"is_settled": 1}) is True
    assert contract_payload_is_settled({"is_expired": 1}) is True
    assert contract_payload_is_settled({"status": "won"}) is True
    assert contract_payload_is_settled({"status": "open"}) is False
    assert contract_payload_is_settled({"status": "LOST"}) is True
    assert contract_payload_is_settled({"status": "EXPIRED"}) is True
