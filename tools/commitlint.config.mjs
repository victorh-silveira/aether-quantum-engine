export default {
  // ---- CONFIGURAÇÃO BASE ----
  extends: ["@commitlint/config-conventional"], // Utiliza as normas convencionais (Angular/Conventional Commits)

  rules: {
    // ---- TIPOS DE COMMIT (type-enum) ----
    "type-enum": [
      2, "always",
      [
        "build",    // Alterações no sistema de build ou dependências externas
        "chore",    // Tarefas de manutenção que não alteram código ou testes
        "ci",       // Mudanças em arquivos e scripts de configuração de CI/CD
        "docs",     // Somente mudanças na documentação
        "feat",     // Uma nova funcionalidade (feature)
        "fix",      // Correção de um erro (bug fix)
        "perf",     // Mudança de código focada em melhoria de performance
        "qa",       // Relacionado a Garantia de Qualidade/Processos de homologação
        "refactor", // Mudança de código que não corrige bug nem adiciona funcionalidade
        "revert",   // Reversão de um commit anterior
        "style",    // Mudanças que não afetam o sentido do código (espaços, formatação)
        "test",     // Adição ou correção de testes existentes
      ],
    ],

    // ---- ESCOPOS DO PROJETO (scope-enum) ----
    "scope-enum": [
      2, "always",
      [
        "all", "api", "app", "config", "deps", "domain", "engine", 
        "infra", "llm", "orchestrator", "pres", "release", "repo", 
        "risk", "scripts", "test", "tools", "ws"
      ],
    ],

    // ---- REGRAS DE FORMATAÇÃO E OBRIGATORIEDADE ----
    "type-case": [2, "always", "lower-case"],     // O tipo deve ser sempre em letras minúsculas
    "type-empty": [2, "never"],                   // O tipo é obrigatório (não pode ser vazio)
    "scope-empty": [2, "never"],                  // O escopo é obrigatório neste projeto
    "subject-empty": [2, "never"],                // O assunto (título) do commit é obrigatório
    "subject-case": [0],                          // Desativa a validação de case para o assunto
    "body-leading-blank": [2, "always"],          // Deve haver uma linha em branco antes do corpo
    "body-empty": [2, "never"],                   // O corpo do commit é obrigatório (descrição detalhada)
    "header-max-length": [2, "always", 100],      // Limite de 100 caracteres para a primeira linha
  },
};