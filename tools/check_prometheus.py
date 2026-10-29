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

side_names = {"1": "CALL", "-1": "PUT", "0": "NEUTRO"}

print("=========================================================================================")
print("  ESTADO DOS 12 ATIVOS OTC NO PROMETHEUS (PERIODO M15 / 900s)")
print("=========================================================================================")
print(f"{'SIMBOLO':<12} | {'STATUS M15':<18} | {'SINAL (SIDE)':<14} | {'FRESH':<8} | {'PRECO ATUAL':<12}")
print("-" * 75)
for s in sorted(val.keys()):
    v_val = val[s]
    v_desc = "OPERACIONAL (1)" if v_val == "1" else f"BLOQUEADO ({v_val})"
    s_val = side.get(s, "0")
    s_desc = side_names.get(s_val, s_val)
    f_val = fresh.get(s, "0")
    c_val = close.get(s, "N/A")
    print(f"{s:<12} | {v_desc:<18} | {s_desc:<14} | {f_val:<8} | {c_val:<12}")

print("=========================================================================================")
trading_pnl = get_metric("aether_trading_pnl_usd").get("global", "0")
trading_contracts = get_metric("aether_trading_contracts_total").get("global", "0")
balance = get_metric("aether_session_balance_usd").get("global", "0")
profit = get_metric("aether_session_profit_usd").get("global", "0")
print(f"Banca: ${balance} | PnL Trading: ${trading_pnl} | Lucro Sessao: ${profit} | Contratos: {trading_contracts}")
print("=========================================================================================")
