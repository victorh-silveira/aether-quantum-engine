export default {
  // ---- CONFIGURAÇÃO BASE ----
  extends: ["@commitlint/config-conventional"], // Utiliza as normas convencionais (Angular/Conventional Commits)

  plugins: [
    {
      rules: {
        "subject-pt-br": ({ subject }) => {
          if (!subject) return [true];
          // Palavras em ingles proibidas no assunto do commit
          const englishWords = [
            /\badd\b/i, /\badding\b/i, /\badded\b/i,
            /\bupdate\b/i, /\bupdating\b/i, /\bupdated\b/i,
            /\bremove\b/i, /\bremoving\b/i, /\bremoved\b/i,
            /\brefine\b/i, /\brefining\b/i, /\brefined\b/i,
            /\btune\b/i, /\btuning\b/i, /\btuned\b/i,
            /\bveto\b/i, /\bvetoing\b/i, /\bvetoed\b/i,
            /\btighten\b/i, /\btightening\b/i, /\btightened\b/i,
            /\bimplement\b/i, /\bimplementing\b/i, /\bimplemented\b/i,
            /\balign\b/i, /\baligning\b/i, /\baligned\b/i,
            /\bfix\b/i, /\bfixing\b/i, /\bfixed\b/i,
            /\btest\b/i, /\btesting\b/i, /\btested\b/i,
            /\bclean\b/i, /\bcleaning\b/i, /\bcleaned\b/i,
            /\ballow\b/i, /\ballowing\b/i, /\ballowed\b/i,
            /\bchange\b/i, /\bchanging\b/i, /\bchanged\b/i,
          ];
          const found = englishWords.find((regex) => regex.test(subject));
          if (found) {
            return [
              false,
              `O assunto do commit deve ser escrito em Portugues (PT-BR). Palavra em ingles detectada: "${subject}"`,
            ];
          }
          return [true];
        },
      },
    },
  ],

  rules: {
    // ---- REGRA OBRIGATÓRIA DE IDIOMA PT-BR ----
    "subject-pt-br": [2, "always"],

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