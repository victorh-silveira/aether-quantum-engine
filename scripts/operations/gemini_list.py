"""Lista todos os modelos disponíveis na API Google Gemini."""

import os

from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Erro: GEMINI_API_KEY não encontrada no ambiente/.env")
        return

    try:
        client = genai.Client(api_key=api_key)
        print("Conectando à API Gemini...")
        all_models = list(client.models.list())
        models = [m for m in all_models if "gemini" in m.name.lower()]
        print(f"\nEncontrados {len(models)} modelos Gemini (de {len(all_models)} no total):\n")

        for m in sorted(models, key=lambda x: x.name):
            print(f"Modelo: {m.name}")
            print(f"  - Display Name: {getattr(m, 'display_name', 'N/A')}")
            print(f"  - Versão: {getattr(m, 'version', 'N/A')}")
            methods = getattr(m, "supported_methods", [])
            print(f"  - Métodos Suportados: {', '.join(methods) if methods else 'N/A'}")
            print("-" * 50)

    except Exception as e:
        print(f"Erro ao listar modelos: {e}")


if __name__ == "__main__":
    main()
