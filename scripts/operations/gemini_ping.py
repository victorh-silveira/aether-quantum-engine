"""Diagnostico de conexao com a API Google Gemini."""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3.1-pro-preview"


def banner(msg):
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def test_api_key():
    banner("1. Verificando GEMINI_API_KEY")
    if not API_KEY:
        print("  [FALHA] GEMINI_API_KEY nao encontrada no ambiente/.env")
        return False
    masked = API_KEY[:6] + "..." + API_KEY[-4:]
    print(f"  [OK] Chave encontrada: {masked}  (len={len(API_KEY)})")
    return True


def test_import():
    banner("2. Importando google-genai SDK")
    try:
        print(f"  [OK] google-genai importado com sucesso (version: {getattr(genai, '__version__', '?')})")
        return True
    except Exception as e:
        print(f"  [FALHA] Erro ao importar: {e}")
        return False


def test_client():
    banner("3. Criando cliente Gemini")
    try:
        client = genai.Client(api_key=API_KEY)
        print("  [OK] Cliente criado com sucesso")
        return client
    except Exception as e:
        print(f"  [FALHA] Erro ao criar cliente: {e}")
        return None


def test_list_models(client):
    banner("4. Listando TODOS os modelos disponiveis na conta")
    try:
        models = list(client.models.list())
        print(f"  [OK] {len(models)} modelos encontrados.")
        print("  Lista completa de IDs:")
        for m in sorted(models, key=lambda x: x.name):
            tag = " <<< CONFIGURADO" if MODEL in m.name else ""
            methods = getattr(m, "supported_methods", [])
            can_gen = "generateContent" in str(methods)
            gen_tag = " [Suporta GenerateContent]" if can_gen else " [NAO Suporta]"
            print(f"    - {m.name}{gen_tag}{tag}")

        model_names = [m.name for m in models]
        return any(MODEL in n for n in model_names)
    except Exception as e:
        print(f"  [FALHA] Erro ao listar: {e}")
        return False


def test_generate(client):
    banner("5. Teste de geracao (ping)")
    prompt = "Responda apenas com a palavra CALL ou PUT. Escolha uma."
    print(f"  Modelo: {MODEL}")
    print(f"  Prompt: {prompt}")
    print("  Enviando...")

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        elapsed = (time.time() - t0) * 1000

        text = ""
        if response and response.candidates:
            content = response.candidates[0].content
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

        if text.strip():
            print(f"  [OK] Resposta recebida em {elapsed:.0f}ms")
            print(f"  Resposta: '{text.strip()}'")
            if elapsed < 2000:
                print("  [AVALIAÇÃO] Desempenho EXCELENTE! (Latência < 2s)")
            elif elapsed < 5000:
                print("  [AVALIAÇÃO] Desempenho BOM! (Latência < 5s)")
            else:
                print("  [AVALIAÇÃO] Desempenho LENTO! (Latência > 5s). Pode haver gargalo na rede ou na API.")
            return True
        print(f"  [AVISO] Resposta vazia em {elapsed:.0f}ms")
        print(f"  Response object: {response}")
        if hasattr(response, "candidates") and response.candidates:
            c = response.candidates[0]
            print(f"  Finish reason: {getattr(c, 'finish_reason', '?')}")
            if hasattr(c, "safety_ratings"):
                print(f"  Safety ratings: {c.safety_ratings}")
        return False
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(f"  [FALHA] Erro apos {elapsed:.0f}ms: {type(e).__name__}: {e}")
        return False


def test_generate_with_system(client):
    banner("6. Teste com system instruction (modo soberano)")
    system = "Voce e um analista de mercado financeiro. Analise os dados e responda APENAS com CALL ou PUT."
    prompt = (
        "RSI M1=55, RSI M5=62, tendencia H1=alta, M15=lateral. "
        "O preco esta acima da EMA21. Qual sua decisao? Responda CALL ou PUT."
    )
    print(f"  System: {system[:60]}...")
    print(f"  Prompt: {prompt[:60]}...")

    t0 = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=256,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        elapsed = (time.time() - t0) * 1000
        text = ""
        if response and response.candidates:
            content = response.candidates[0].content
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if hasattr(part, "text") and part.text:
                    text += part.text

        if text.strip():
            print(f"  [OK] Resposta em {elapsed:.0f}ms: '{text.strip()}'")
            upper = text.strip().upper()
            is_valid = upper in ("CALL", "PUT")

            if is_valid:
                print(f"  [OK] Direcao valida recebida: {upper}")
            else:
                print(f"  [AVISO] Resposta nao e CALL/PUT puro: '{text.strip()}'")

            if elapsed < 2000 and is_valid:
                print("  [AVALIAÇÃO] Modelo respondeu RÁPIDO e CORRETAMENTE. Modelo EXCELENTE para operações!")
            elif elapsed < 5000 and is_valid:
                print("  [AVALIAÇÃO] Modelo respondeu CORRETAMENTE num tempo aceitável. Modelo BOM para operações!")
            elif not is_valid:
                print(
                    "  [AVALIAÇÃO] Modelo FALHOU em seguir as instruções (não respondeu apenas CALL/PUT). NÃO RECOMENDADO para operações estritas!"
                )
            else:
                print("  [AVALIAÇÃO] Modelo LENTO. Cuidado ao usar em operações de alta frequência.")

            return is_valid
        print(f"  [AVISO] Resposta vazia em {elapsed:.0f}ms")
        return False
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(f"  [FALHA] Erro apos {elapsed:.0f}ms: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  AETHER QUANTUM ENGINE - Diagnostico Gemini API")
    print(f"  Modelo alvo: {MODEL}")
    print("=" * 60)

    results = {}

    results["api_key"] = test_api_key()
    if not results["api_key"]:
        print("\n>>> Sem API key, impossivel continuar.")
        sys.exit(1)

    results["import"] = test_import()
    if not results["import"]:
        sys.exit(1)

    client = test_client()
    results["client"] = client is not None
    if not client:
        sys.exit(1)

    results["list_models"] = test_list_models(client)
    results["generate"] = test_generate(client)
    results["generate_system"] = test_generate_with_system(client)

    banner("RESUMO")
    all_ok = True
    for k, v in results.items():
        status = "OK" if v else "FALHA"
        icon = "V" if v else "X"
        print(f"  [{icon}] {k}: {status}")
        if not v:
            all_ok = False

    if all_ok:
        print("\n  >>> TUDO OK! A API Gemini esta funcionando corretamente.")
        print("  >>> O problema pode ser no prompt do bot ou rate limiting.")
    else:
        print("\n  >>> PROBLEMAS DETECTADOS. Verifique os itens com FALHA acima.")

    sys.exit(0 if all_ok else 1)
