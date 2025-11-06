# Schema PDF Extractor

Uma aplicação que combina heurísticas simples e chamadas a modelos de linguagem para converter PDFs orientados por um schema em JSON estruturado. O projeto oferece tanto uma interface visual em Gradio quanto um modo de linha de comando para processar lotes de arquivos.

## Visão Geral

- **Extração padronizada**: um schema JSON descreve os campos esperados e o modelo de linguagem retorna apenas esses campos (ou `null`).
- **Upload em lote**: a UI aceita múltiplos PDFs em uma única execução.
- **CLI em lote**: o modo de linha de comando varre uma pasta e processa todos os PDFs encontrados.
- **Cache inteligente**: PDFs já processados com o mesmo schema voltam instantaneamente, poupando chamadas ao modelo.
- **Prompt consistente**: poucos-shot examples embutidos demonstram o formato esperado, reduzindo erros de mapeamento.

## Pré-Requisitos

- Python 3.11+
- Acesso à API da OpenAI (variável de ambiente `OPENAI_API_KEY`).
- Dependências listadas em `requirements.txt`.

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Executando a Interface Gradio

```bash
python app.py
```

Isso abrirá (ou exibirá no terminal) a URL local da interface. Na UI:

1. Informe o schema JSON (um valor padrão é carregado). Apenas campos presentes no schema serão retornados.
2. Selecione um ou mais arquivos PDF.
3. Clique em **Extrair**. O painel de JSON mostrará apenas os campos `dados` (um objeto por arquivo). O log indica se o resultado veio do cache ou do modelo.

## Utilizando o Modo CLI

É possível processar lotes diretamente pela linha de comando, apontando para uma pasta com PDFs:

```bash
python app.py --cli --pdf-dir "caminho/para/pasta" --schema-file schema.json
```

Opções principais:

- `--cli`: ativa o modo CLI.
- `--pdf-dir`: diretório contendo arquivos `.pdf` (obrigatório no modo CLI).
- `--schema-file`: caminho para um arquivo JSON com o schema desejado (opcional).
- `--schema`: string JSON inline que sobrepõe o arquivo, útil para scripts automáticos.

Saída:

- `stdout`: lista JSON com objetos contendo `arquivo`, `dados` e `status` para cada PDF.
- `stderr`: log textual resumindo o resultado de cada arquivo (útil para monitoramento).

## Desafios e Soluções

| Desafio | Resolução | Resultado |
| --- | --- | --- |
| **Prompt inconsistente gerando campos trocados** | Criação de prompt fixo com exemplos few-shot e instruções explícitas para seguir o schema. | Redução de respostas erradas e valores em campos trocados. |
| **Tempo de resposta quando o mesmo PDF era enviado várias vezes** | Cache baseado em hash do PDF + assinatura do schema. Se o documento e o schema não mudam, a resposta é reutilizada. | Economia de chamadas à API e experiência mais rápida. |
| **Suporte a múltiplos cenários** | Entrada de schema customizável na UI e bandeira de linha de comando para processamento em lote. | Usuários podem adaptar o schema para novos campos e integrar em pipelines automatizados. |
| **Robustez na leitura de PDFs** | Tratamento de erros em `pdfplumber` e mensagens claras quando não há texto extraído ou a leitura falha. | Facilidade para diagnosticar problemas de arquivos corrompidos ou digitalizados. |
| **Consistência da chave JSON do schema** | Serialização com `sort_keys=True` e reutilização da mesma string ao compor o prompt e o cache. | Evita rotação desnecessária de cache e garante prompts determinísticos. |

## Fluxo Interno

1. **Leitura do PDF**: funiliza múltiplas origens (dicionário do Gradio, arquivo, caminho, objeto file-like) em bytes.
2. **Extração de texto**: `pdfplumber` gera linhas normalizadas.
3. **Prompt**: constrói uma mensagem com o schema desejado, instruções e exemplos.
4. **Chamada ao modelo**: `gpt-5-mini` retorna JSON estruturado; falhas são registradas.
5. **Cache**: hash do schema (ordenado) + hash do PDF identifica repetições.
6. **Saída**: UI mostra apenas os objetos `dados`; CLI imprime todos os metadados necessários.

## Estrutura do Projeto

```
app.py             # Aplicação e CLI
requirements.txt   # Dependências Python
```

## Como Estender

- **Novos campos**: basta adicionar ao schema fornecido. O modelo seguirá as chaves dadas.
- **Nova LLM**: troque `gpt-5-mini` e ajuste a chamada na função `call_llm`. Caso precise de outra chave de API, configure via variável de ambiente.
- **Persistência de cache**: atualmente o cache é em memória. Para um ambiente multi-instância, persistir em Redis/Mongo é simples, já que a chave é uma string e o valor é JSON.

## Licença

Projeto mantido para fins experimentais. Adapte conforme necessário para seu caso de uso.
