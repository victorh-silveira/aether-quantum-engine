import urllib.request
import json

def get_metric(metric):
    url = f"http://localhost:9090/api/v1/query?query={metric}"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode())
            return {
                r["metric"].get("symbol", "global"): r["value"][1]
                for r in data["data"]["result"]
            }
    except Exception:
        return {}

val = get_metric("aether_indicator_validation%7Bperiod=%22900%22%7D")
side = get_metric("aether_indicator_side%7Bperiod=%22900%22%7D")
fresh = get_metric("aether_indicator_fresh%7Bperiod=%22900%22%7D")
close = get_metric("aether_indicator_close%7Bperiod=%22900%22%7D")

NAMES = {
    "OTC_AEX": "Netherlands 25",
    "OTC_AS51": "Australia 200",
    "OTC_DJI": "Wall Street 30",
    "OTC_FCHI": "France 40",
    "OTC_FTSE": "UK 100",
    "OTC_GDAXI": "Germany 40",
    "OTC_HSI": "Hong Kong 50",
    "OTC_N225": "Japan 225",
    "OTC_NDX": "US Tech 100",
    "OTC_SPC": "US 500",
    "OTC_SSMI": "Swiss 20",
    "OTC_SX5E": "Euro 50",
}

side_names = {"1": "CALL", "-1": "PUT", "0": "NEUTRO"}

print("====================================================================================================")
print("  ESTADO DOS 12 ATIVOS OTC NO PROMETHEUS (PERIODO M15 / 900s)")
print("====================================================================================================")
print(f"{'NOME DO ATIVO (DERIV)':<20} | {'SIMBOLO':<10} | {'STATUS M15':<18} | {'SINAL (SIDE)':<14} | {'FRESH':<6} | {'PRECO':<10}")
print("-" * 100)
for s in sorted(val.keys()):
    name = NAMES.get(s, s)
    v_val = val[s]
    v_desc = "OPERACIONAL (1)" if v_val == "1" else f"BLOQUEADO ({v_val})"
    s_val = side.get(s, "0")
    s_desc = side_names.get(s_val, s_val)
    f_val = fresh.get(s, "0")
    c_val = close.get(s, "N/A")
    print(f"{name:<20} | {s:<10} | {v_desc:<18} | {s_desc:<14} | {f_val:<6} | {c_val:<10}")

print("====================================================================================================")
trading_pnl = get_metric("aether_trading_pnl_usd").get("global", "0")
trading_contracts = get_metric("aether_trading_contracts_total").get("global", "0")
balance = get_metric("aether_session_balance_usd").get("global", "0")
profit = get_metric("aether_session_profit_usd").get("global", "0")
print(f"Banca: ${balance} | PnL Trading: ${trading_pnl} | Lucro Sessao: ${profit} | Contratos: {trading_contracts}")
print("=========================================================================================")
