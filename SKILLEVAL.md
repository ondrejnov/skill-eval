# skill-eval – spouštěč evalů skillů/pluginů mimo Claude Code

Specifikace pro AI kódovacího agenta. Cíl: samostatná Python aplikace v `/var/www/skill-eval`, která čte eval sady ve
formátu **Claude Code plugin evals** (<https://code.claude.com/docs/en/plugin-evals>), spouští agenta nad skilly a MCP
servery přes **OpenRouter** (LangGraph/LangChain) a vyhodnocuje výsledky stejně jako `claude plugin eval`.

## 0. Proč a co řeší

`claude plugin eval skills/webstore-core --case zapis-bez-dotazu --runs 1 --ablation none --mocks off --allow-tools 'mcp__plugin_webstore-core_webstore__*' --model haiku`
funguje, ale:

| problém v Claude Code | jak to řeší skill-eval |
|---|---|
| vázané na Anthropic providera / předplatné | libovolný model přes OpenRouter (`OPENROUTER_API_KEY`), agent i soudce zvlášť |
| MCP se bere jen z `<plugin>/.mcp.json`, `${VAR}` se neexpanduje (token natvrdo, jinak 401) | `.mcp.json` s expanzí `${VAR}` z env + přepis serverů v `skilleval.toml` / `--mcp-config` |
| plugin = jeden skill; skilly `webstore-*` se navzájem odkazují (core → orders …) | víc pluginů/skillů v jednom běhu (`--plugin` opakovaně, pole `plugins`) |
| zinscenované situace (conflict, chyba dopravce, rights 5) nejdou s reálným serverem nasimulovat | rozšíření `skilleval.yaml` vedle case: fault injection na N-té volání, MCP profil (token jiného uživatele), setup/teardown |
| název toolů `mcp__plugin_<skill>_<server>__*` musí sedět v `--allow-tools` | stejné pojmenování (kvůli kompatibilitě graderů), ale grant je výchozí pro servery pluginu |
| zápisové evaly mění dev DB | volitelné `setup`/`teardown` hooky (opt-in `--hooks`) |

**Nekompatibilní změny formátu jsou zakázané.** Každá sada, která projde v `claude plugin eval`, musí jít spustit
v skill-eval bez úprav a naopak – rozšíření žijí jen v souborech, které Claude Code nečte (`skilleval.yaml`,
`skilleval.toml`), nikdy jako nové klíče v `prompt.md`/`case.yaml`/graderech/mockách (Claude Code u nich hlásí chybu).

## 1. Technologie a struktura

- Python 3.13, Poetry (venv v `.venv/` projektu), CLI `skilleval` (entry point v `pyproject.toml`), `typer` nebo `argparse`.
- Agent: **LangGraph** (`StateGraph` agent ↔ tools, vlastní – ne `create_react_agent`, potřebujeme kontrolu nad počítáním
  tahů, timeoutem, trace a fault injection). Model `langchain_openai.ChatOpenAI(base_url=OPENROUTER_BASE_URL, api_key=…)`.
- MCP klient: oficiální `mcp` SDK (`streamablehttp_client`, `stdio_client`, `sse_client`); `langchain-mcp-adapters` jen
  pokud nebrání fault injection a vlastnímu pojmenování toolů – jinak vlastní tenký adaptér MCP tool → LangChain `StructuredTool`.
- Další: `pydantic` v2 (modely souborů a výsledků), `pyyaml`, `jinja2` (report), `httpx`, `python-dotenv`, `pytest`, `ruff`.
- Asynchronní běh (`asyncio`), souběh `-j`.

```
/var/www/skill-eval/
  SKILLEVAL.md               tato specifikace (neměnit bez dohody)
  README.md                  instalace, použití, příklady
  pyproject.toml, .env.example, skilleval.toml.example
  skilleval/
    cli.py                   příkazy run / list / validate / report / init --bare
    config.py                .env, skilleval.toml, aliasy modelů, ceny
    suite/                   načtení sady: discovery.py, models.py (pydantic), loader.py, validate.py
    plugin.py                nalezení pluginu, katalog skillů, MCP konfigurace pluginu
    harness/                 emulace Claude Code: prompt.py (system prompt), tools_builtin.py (Skill, Read, Glob, Grep,
                             Write, Edit, TodoWrite, AskUserQuestion), sandbox.py (workspace, HOME), permissions.py
    mcp/                     client.py (připojení, tools/list, instructions), naming.py, mocks.py (fixed/agent/replay),
                             faults.py (rozšíření), recorder.py
    agent/                   graph.py (LangGraph), model.py (OpenRouter, aliasy), trace.py (stream-json kompatibilní)
    grading/                 regex.py, tool_used.py, tool_order.py, file_exists.py, llm.py, baseline.py, targets.py, score.py
    runner.py                orchestrace: cases × arms × runs, concurrency, timeout, cost ceiling, exit kódy
    results/                 aggregate.py (aggregate-result.json), report.py + templates/report.html.j2, summary.py (tabulka)
  tests/                     pytest (FakeModel, fake MCP server, ukázkový plugin v tests/fixtures/)
```

## 2. Konfigurace

### 2.1 Env

Pořadí (pozdější přebíjí): `/var/www/skill-eval/.env` → `--env-file <path>` (opakovatelně) → proměnné procesu.
`skilleval.toml` může přes `env_files` odkazovat na další soubory – odtud se bere
`OPENROUTER_API_KEY` (a případně `OPENROUTER_BASE_URL`). Klíče se nikdy nelogují ani nezapisují do výsledků/reportu
(maskovat `Authorization`, `*_TOKEN`, `*_KEY`, `*PASSWORD*`).

| proměnná | význam |
|---|---|
| `OPENROUTER_API_KEY` | povinná |
| `OPENROUTER_BASE_URL` | default `https://openrouter.ai/api/v1` |
| `SKILLEVAL_MODEL`, `SKILLEVAL_JUDGE_MODEL` | výchozí modely (jinak z `skilleval.toml`) |
| `WEBSTORE_MCP_TOKEN` (příklad) | cokoli, na co odkazuje `${VAR}` v `.mcp.json` / `skilleval.toml` |

### 2.2 `skilleval.toml`

Hledá se v CWD, pak v kořeni pluginu, pak `/var/www/skill-eval/skilleval.toml`; `--config` přebíjí.

```toml
env_files = [".env.local"]

[models]                       # aliasy jako v Claude Code (--model haiku) → OpenRouter slug
default = "haiku"
judge = "sonnet"
[models.aliases]
haiku  = "anthropic/claude-haiku-5.5"
sonnet = "anthropic/claude-sonnet-5.5"
opus   = "anthropic/claude-opus-5.5"
[models.options]               # volitelné per model: reasoning effort, max_tokens, temperature
"anthropic/claude-haiku-5.5" = { reasoning_effort = "low" }

[mcp.servers.webstore]         # přepis/doplnění serveru z .mcp.json pluginu (merge po klíčích)
type = "http"
url = "${WEBSTORE_MCP_URL:-http://127.0.0.1:8890/api/mcp}"
headers = { Authorization = "Bearer ${WEBSTORE_MCP_TOKEN}" }

[mcp.profiles.limited]         # rozšíření: profil = jiné hlavičky/env (např. token uživatele s rights 5)
servers.webstore.headers = { Authorization = "Bearer ${WEBSTORE_MCP_TOKEN_RIGHTS5}" }

[pricing]                      # fallback, když OpenRouter nevrátí usage.cost
"anthropic/claude-haiku-5.5" = { input = 1.0, output = 5.0 }   # USD / 1M tokenů
```

Slugy modelů ověř proti `GET {OPENROUTER_BASE_URL}/models` (příkaz `skilleval models` je vypíše/ověří aliasy).
Expanze `${VAR}` a `${VAR:-default}` platí v `.mcp.json` i `skilleval.toml`; nedefinovaná proměnná bez defaultu = chyba
konfigurace s názvem proměnné (ne tichý prázdný řetězec → 401).

## 3. Načtení sady (kompatibilita s plugin evals)

Implementuj **přesně** sekci „Eval suite reference“ dokumentace. Shrnutí, které musí sedět:

- **Target**: adresář pluginu (`.`), jeden `prompt.md`/`case.yaml`, nebo vynechaný (= CWD). Instalované pluginy
  (`name@marketplace`) nepodporovat – srozumitelná chyba.
- **Plugin**: adresář s `plugin.json` nebo `.claude-plugin/plugin.json` (název z manifestu), nebo *skills-directory plugin*:
  adresář se `SKILL.md` (= plugin s jedním skillem, název = název adresáře, jako `skills/webstore-core`), případně adresář se
  `skills/<x>/SKILL.md`. Eval adresář `evals/` nebo `experimental.evals` z manifestu nebo `--eval-dir` (relativní, bez `..`).
- **Case** = adresář s `prompt.md` a/nebo `case.yaml`; skupiny = vnoření pod ne-case adresář; `results/` a `mocks/` nejsou case.
  Case bez graderu = chyba načtení (`invalid case.yaml … graders`).
- **prompt.md frontmatter** (neznámý klíč = chyba): `schema_version, name, description, tags, plugins, runs (1–50),
  expected_outcome, model, max_turns (default 10, ≤200), timeout_seconds (default 300, ≤3600), allowed_tools,
  append_system_prompt, env` (klíče jen `EVAL_[A-Z0-9_]*`). Tělo = prompt doslova (`@path` se neexpanduje).
- **case.yaml**: povinně `schema_version: "1.1"` a `name`; top-level `description, tags, plugins, runs, expected_outcome`;
  `execution.{model, max_turns, timeout_seconds, allowed_tools, append_system_prompt, env, prompt}`;
  `context.{scaffold_script, history_file, add_dirs}`; `graders: [{name, type, …, criteria}]`.
  Sloučení: frontmatter `prompt.md` přebíjí `case.yaml`, tělo `prompt.md` je prompt, `graders/*.md` se přidají za graders z yaml.
  Neznámé klíče v `case.yaml` (např. `expected_outcome` je známý; `source` apod.) → chyba jako v Claude Code.
- **Grader soubor** `graders/<name>.md`: frontmatter `type` (povinný), `weight` (default 1, >0), `arm` (`with-only|both`)
  + volby typu; tělo = `criteria` u `llm`/`baseline`, u `regex` může být `pattern` ve frontmatteru.
- **Mocks**: `evals/mocks/<server>/<tool>.md` (celá sada) a `<case>/mocks/…` (přebíjí soubor po souboru), `_server.md`,
  `_tools.json`, `fixtures/`, `.replay/<server>/`.
- Ostatní soubory v case adresáři (`expected.md`, `skilleval.yaml`, fixtures) se ignorují, pokud na ně nic neodkazuje.

`skilleval validate <target>` = jen načte a zvaliduje sadu (včetně regex dialektu v `expect:`), vypíše chyby s cestou, exit 1.

## 4. Emulace běhu Claude Code (harness)

Cíl: agent se má chovat co nejpodobněji `claude -p` s načteným pluginem, aby výsledky byly srovnatelné.

### 4.1 Izolace běhu
- Každý run: nový dočasný `workspace/` (CWD agenta) a `home/`; `--keep-temp` je ponechá a vypíše cestu.
- Nenačítá se nic z uživatele/projektu (žádný CLAUDE.md, žádné `.mcp.json` mimo plugin, žádná paměť).
- Agent nesmí číst eval adresář (prompt, gradery, mocky, `expected.md`) – Read/Glob/Grep to musí odmítnout.
- Čitelné: workspace, adresáře pluginů (kvůli `references/` skillů), `context.add_dirs` (read-only).
- Zapisovatelné (jen s grantem `Write`/`Edit`): workspace.
- `scaffold_script` jen s `--scaffold`: `bash <script>` v prázdném workspace, env = `PATH`, `HOME`=temp home, `TMPDIR`,
  `TERM=dumb`; limit 120 s; nenulový exit → run skóre 0, chyba `scaffold failed`.
- `env` case (`EVAL_*`) + allowlist (`PATH`, locale, proxy/certifikáty) se předají MCP serverům typu stdio a hookům.

### 4.2 System prompt
Šablona `harness/prompt.py` (anglicky, jako Claude Code): role coding/agent asistenta v neinteraktivním režimu, CWD, datum,
seznam dostupných toolů, pravidla pro skilly a **instrukce MCP serverů** z `initialize.instructions` (Claude Code je vkládá –
webstore server je posílá). Na konec `append_system_prompt` z case. Katalog skillů ve stylu Claude Code:
`<available_skills>` s položkami `name` (`<plugin>:<skill>` i krátký název) + `description` z frontmatteru `SKILL.md`.
Text šablony drž v jednom souboru, ať jde porovnávat/ladit.

### 4.3 Vestavěné tooly
Názvy a vstupy jako Claude Code (kvůli graderům `tool_used`/`tool_order` a regexům nad trace):

| tool | vstup | chování |
|---|---|---|
| `Skill` | `{"skill": "<name>", "args"?: str}` | vrátí tělo `SKILL.md` (bez frontmatteru) + `Base directory for this skill: <abs path>`; neznámý skill = tool error. Přijímá `name` i `plugin:name` |
| `Read` | `{file_path, offset?, limit?}` | čtení s čísly řádků (cat -n), jen povolené cesty |
| `Glob` | `{pattern, path?}` | |
| `Grep` | `{pattern, path?, glob?, output_mode?, -i?}` | Python `re` |
| `Write`, `Edit` | jako Claude Code | jen s grantem, jen workspace |
| `TodoWrite` | `{todos: [...]}` | no-op, vrací OK |
| `AskUserQuestion` | `{questions: [...]}` | neinteraktivně: vrátí „User is not available in this non-interactive session; proceed with your best judgement or stop and explain.“ (rozšíření `simulated_user` viz 7) |
| `Agent` | – | v1 nepodporovat; vyžádaný = vypsat jako `not supported` |
| `Bash`, `WebFetch`, `WebSearch` | – | v1 nepodporovat (bez sandboxu je nespouštět); grant = varování `not supported`, tool se nenabídne |

Granty: tool je dostupný, když je v `allowed_tools` case a patří do read-only množiny (`Read, Glob, Grep, NotebookRead,
Skill, AskUserQuestion, Agent, TodoWrite, Task*`), nebo je v `--allow-tools`. Pozor na odchylku: v Claude Code je `Skill`
dostupný jen když ho case uvede; to zachovej (webstore cases `allowed_tools` neuvádějí → zjisti chování CC empiricky
a zdokumentuj v README; pokud CC skill i tak nabízí, nabízej taky).

### 4.4 MCP tooly
- Servery pluginu z `<plugin>/.mcp.json` (případně `mcpServers` v manifestu) sloučené s `[mcp.servers]` ze `skilleval.toml`
  a `--mcp-config <json>`; typy `http` (Streamable HTTP), `sse`, `stdio`.
- Pojmenování **přesně jako Claude Code**: `mcp__plugin_<plugin>_<server>__<tool>` (zachovej znaky jako CC, např. pomlčky
  v názvu pluginu). Modelu se tool nabízí pod tímto jménem, description a `inputSchema` z `tools/list` (stránkování `cursor`).
- Režimy podle `--mocks` (viz 5): bez mocku a bez `--mocks off`/`--allow-real-servers` se reálný server nespouští.
- Grant: tooly reálných serverů pluginu jsou **povolené automaticky** (rozdíl proti CC, který vyžaduje `--allow-tools`);
  `--strict-grants` zapne chování CC. `--allow-tools` přijímá globy (`mcp__plugin_webstore-core_webstore__*`).
- Výsledek toolu pro model: textové bloky `content` spojené (jako CC); `structuredContent` jen když chybí text;
  `resource_link` → text `[resource] <uri> (<name>)`; `isError` → tool error. Výsledek > `--max-tool-result-chars`
  (default 100 000) zkrátit s poznámkou.
- Spojení na server = jedno na run (ne sdílené mezi runy – izolace), `initialize` + `tools/list` cachovat na úrovni běhu
  sady jen pro katalog, ne pro volání.

### 4.5 Smyčka agenta (LangGraph)
- Stav: zprávy, počet tahů, náklady, trace. Uzel `agent` (model s `bind_tools`) → `tools` (paralelní tool calls pořadím
  jako v odpovědi) → `agent` …; konec, když model nevolá tool.
- **Tah** = jedno volání modelu. Dosažení `max_turns` → run error `max turns reached (N)`, run se přesto hodnotí.
- `timeout_seconds` (wall clock) → zrušit, error `timed out after Ns`, hodnotit, co vzniklo.
- HTTP chyba API modelu (429, 402, 5xx po retry) → run **skóre 0 bez hodnocení**, error s textem chyby, exit ≠ 0
  (jako „ended on an API error“). Retry: 3× s backoffem na 429/5xx/timeouty spojení.
- OpenRouter: `extra_body={"usage": {"include": True}}` → `usage.cost` pro přesné náklady; jinak výpočet z `[pricing]`.
  `reasoning.effort` z `[models.options]`. U Anthropic modelů zapnout prompt caching (`cache_control` na system prompt
  a definice toolů), je-li to přes OpenRouter podporováno – šetří náklady při opakovaných runech.
- `context.history_file` (`.jsonl` transcript CC) → převést na LangChain zprávy a prompt připojit jako další user tah;
  takový case běží jen with-arm, pokud není `--ablation with-without`.

### 4.6 Trace (kompatibilní s CC)
Každý run zapíše `results/<ts>/traces/<case>/<arm>-<n>.jsonl` ve formátu **stream-json `claude -p`** (jedna zpráva na řádek):
`{"type":"system","subtype":"init",…}`, `{"type":"assistant","message":{"role":"assistant","content":[{"type":"text",…},
{"type":"tool_use","id","name","input"}]}}`, `{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id",
"content","is_error"}]}}`, `{"type":"result",…}`. Nad touto podobou běží `regex target: trace` (JSON-escapované uvozovky
jako v CC) a `llm focus: trace` (prvních 12 + posledních 12 řádků). `tracePath` v JSON výsledku.

## 5. Mocky MCP (dle dokumentace)

- `--mocks record` (default): reálné servery se nestartují; tool s mock souborem odpovídá z něj a nepotřebuje grant; tool bez
  mocku není dostupný; server bez mocků = v progress řádku `plugin_<plugin>_<server>[not started: no mock]`.
- `--allow-real-servers`: servery/tooly bez mocku jdou na reálný server, mockované dál z mocků (hybrid, po toolech).
- `--mocks off`: mocky ignorovat, všechno reálně.
- `type: fixed` (default): tělo s náhradami `{{input.<pole>}}` (dot-path) a `{{file:fixtures/{input.<pole>}.json}}`;
  `error: true` → tool error.
- `expect:` (dot-path → `string|number|boolean|array|object`, `/regex/` v omezeném dialektu, literál, seznam literálů);
  porušení → abort runu, skóre 0, `aborted: {server, tool, reason}`, `error: null`.
- `type: agent`: odpovídá judge model podle instrukcí v těle, vidí předchozí volání téhož serveru jako historii;
  `abort_when` = jediné podmínky pro abort; společný rozpočet volání = 4 × `max_turns`, překročení = abort
  `mock call budget exceeded`. `_server.md` s `tools: [...]` (agent pro víc toolů; `<tool>.md` má přednost).
- `_tools.json` = uložený `tools/list` → reálné description/schéma mockovaných toolů; jinak permisivní schéma
  `{"type":"object","additionalProperties":true}`. `skilleval mocks snapshot <server>` uloží `_tools.json` z reálného serveru.
- Záznamy agent mocků z čistých runů do `results/<ts>/mock-recordings/` + `ADOPT.txt`; `.replay/<server>/` odpovídá na
  identické volání (kanonický JSON vstupu) bez volání modelu.
- `target: mock_calls` / `focus: mock_calls`: seznam `{server, tool, input, output, is_error}`.

## 6. Hodnocení (dle dokumentace)

| typ | volby | projde, když |
|---|---|---|
| `regex` | `pattern`, `flags` (jen JS flagy; `i`,`m`,`s` → Python), `match` (`contains` default / `not_contains` / `count:N`), `target` | vzor nalezen v targetu; JS syntaxi převeď (`(?<name>` ok, `\d` ok; `(?i)` = chyba načtení) |
| `tool_used` | `tool`, `input_match` (regex nad JSON-serializovaným vstupem, `json.dumps(..., ensure_ascii=False, separators=(",", ":"))` – ověř proti CC trace, jak serializuje), `min` (1), `max` (∞) | počet volání v intervalu |
| `tool_order` | `before`, `after` (název nebo `{tool, input_match}`) | oba volány a první `before` před prvním `after` |
| `file_exists` | `path` (glob), `exists` (true) | soubor vytvořený během runu (ne scaffoldem, ne jen upravený) |
| `llm` | `criteria` (tělo), `focus` | soudce 3× hlasuje, PASS ve ≥ 2 ze 3 |
| `baseline` | `baseline_file` (.jsonl v case), `criteria` | soudce: run splňuje kritéria aspoň tak dobře jako referenční transcript |

- `target`/`focus`: `last_message` (default), `trace`, `files` (seznam vytvořených cest), `{source: file, path}` (obsah; obrázky
  PNG/JPEG/GIF/WebP soudci jako image, jiné binární = grader fail s vysvětlením), `mock_calls`.
- **Soudce**: model `--judge-model` (alias), structured output `{verdict: "PASS"|"FAIL", explanation, evidence}`; prompt
  česky/anglicky neutrální, rubrika doslova, evidence = výřez, který soudce viděl (ukládá se do výsledku). 3 hlasy paralelně,
  `temperature` default modelu. Náklady soudce zvlášť (`judgeCostUsd`).
- **Skóre runu** = vážený podíl prošlých *scored* graderů; case = průměr runů; case prošel, když with-arm ≥ `--threshold` (1.0).
- **Ablace**: `--ablation with-without` (default, kromě výjimek) / `none`. Without-arm = žádný plugin: žádné skilly, žádné MCP
  servery pluginu (stejně jako v CC – MCP je součást pluginu). `scored: false` v obou armech pro: `tool_used tool: Skill`,
  `regex target: mock_calls` / `llm focus: mock_calls` když všechny mockované servery patří pluginu, `arm: with-only`;
  `arm: both` vynutí; když by byly vyloučené všechny, hodnotí se normálně; při `--ablation none` se nevylučuje nic.
- Single-arm případy: `--ablation none`, `history_file` u path targetu, nenalezený plugin (`single-arm (no Δ)` na stderr).

## 7. Rozšíření (jen skill-eval, Claude Code je ignoruje)

Soubor `<case>/skilleval.yaml` (volitelný; validovaný, neznámý klíč = chyba):

```yaml
mcp_profile: limited              # profil z [mcp.profiles] – např. token uživatele s rights 5
faults:                           # fault injection nad REÁLNÝM serverem (i při --mocks off)
  - tool: orders_update           # název MCP toolu bez prefixu (nebo plný mcp__plugin_…)
    call: 1                       # kolikáté volání (1-based); nebo `calls: [1,3]`, `every: true`
    when: { "dry_run": false }    # volitelně: podmnožina vstupu, která musí sedět
    error: true
    body: |                       # odpověď místo serveru (náhrady {{input.*}} jako u mocků)
      {"error":{"code":"conflict","message":"Záznam byl mezitím změněn (version 12 → 13)."}}
    passthrough_after: true       # další volání jdou na reálný server
  - tool: orders_update
    before_call: 1                # side-effect před N-tým voláním: spusť akci (např. změna záznamu „v legacy“)
    run: ["bash", "fixtures/touch_order.sh"]   # jen s --hooks
setup:    ["bash", "fixtures/reset_71781.sh"]   # před každým runem, jen s --hooks, timeout 120 s, nenulový exit = run error
teardown: ["bash", "fixtures/reset_71781.sh"]   # po každém runu (i po chybě)
simulated_user:                   # odpovědi na AskUserQuestion / na koncovou otázku agenta (max N dalších tahů)
  max_replies: 1
  replies: ["Chci dobropis."]     # nebo `agent: "<instrukce pro judge model, jak odpovídat>"`
tags_extra: [db-write]            # tagy jen pro skill-eval (filtrovatelné --tag)
```

- `--hooks` je opt-in jako `--scaffold` (spouští kód mimo agenta). `setup`/`teardown`/`run` dostanou env
  `SKILLEVAL_RUN_DIR`, `SKILLEVAL_CASE`, `SKILLEVAL_ARM`, `SKILLEVAL_RUN` + env z `env_files` (např. DB přístupy dev DB).
- Fault/simulated_user/profil se projeví v trace (`{"type":"system","subtype":"skilleval_fault",…}`) a v reportu, aby bylo vidět,
  že odpověď nebyla od serveru.
- Bez `skilleval.yaml` se case chová přesně jako v CC.

## 8. CLI

```
skilleval run [TARGET] [--case GLOB]... [--tag T]... [--eval-dir DIR] [--plugin DIR]...
              [--runs N] [-j N] [--model M] [--judge-model M] [--ablation none|with-without]
              [--threshold 0..1] [--max-cost-usd X] [--allow-tools T...] [--strict-grants]
              [--mocks record|off] [--allow-real-servers] [--scaffold] [--hooks]
              [--mcp-config FILE] [--mcp-profile NAME] [--config FILE] [--env-file FILE]...
              [--json [PATH.json]] [--output-dir DIR] [--keep-temp] [--verbose] [--no-report]
skilleval list [TARGET]            # cases, tagy, gradery, potřebné tooly/servery, rozšíření
skilleval validate [TARGET]
skilleval report RESULTS_DIR       # znovu vygeneruje report.html z aggregate-result.json + traces
skilleval compare A.json B.json    # rozdíly skóre po cases (regrese skillu/modelu)
skilleval models                   # ověří aliasy proti OpenRouter /models, vypíše ceny
skilleval mocks snapshot SERVER    # uloží _tools.json z reálného serveru
skilleval init --bare NAME         # šablona case jako `claude plugin eval init --bare`
```

- `--plugin DIR` (opakovatelně) přidá další pluginy/skilly do with-armu (např. `--plugin skills/webstore-orders` k core);
  MCP servery se stejným názvem se slučují (první vyhrává, varování při rozdílné konfiguraci). Pojmenování MCP toolů
  podle pluginu, který server deklaroval.
- Progress: řádek na run `case [arm n/N] score 0.80 · 12 turns · $0.012 · 01-tvrzeni ✓ 02-tvrzeni ✗ …` + `error:` / `mocked:` /
  `not granted:` / `not supported:`. Na konci tabulka `CASE WITH W/OUT Δ RUNS COST NOTES` (nebo `SCORE PASS%` u jednoho armu),
  řádek `N case(s) · mean Δ · čas · $cena` a `Report: <path>`.
- Exit kódy: 0 ok; 1 case pod prahem / chyba načtení / žádné cases / run nešel spustit / API chyba / neplatná volba;
  2 částečný běh (`--max-cost-usd`, auth chyba OpenRouteru hned na začátku) s `partial: true`; 130 přerušeno (zapsat částečné
  výsledky); 143 SIGTERM.
- `--max-cost-usd` se kontroluje před startem každého runu; rozběhnuté doběhnou; nestartované → partial.

## 9. Výstupy

`<eval dir>/results/<timestamp ISO s - místo :>/` (nebo `--output-dir`):

- `aggregate-result.json` – **schéma shodné s CC** (`schemaVersion: 1`, camelCase), minimálně pole, která CC zapisuje
  (vzor: `aggregate-result.json` z běhu `claude plugin eval`):
  `schemaVersion, startedAt, durationSeconds, costUsd, partial, partialReason, suite{root, ablation, modelOverride,
  judgeModel, caseFilter, threshold, concurrency, plugins[{name,path}]}, cases[{name, dir, source, promptMarkdown,
  runsPerCase, timeoutSeconds, maxTurns, graders[{name,type,weight,graderMarkdown,config}], arms{with[], without[]}
  (run: score, passed, turns, costUsd, judgeCostUsd, durationSeconds, startedAt, error, aborted, tracePath,
  skippedPaidGraders, graders[{name, passed, scored, explanation, votes, evidence}]), aggregates{score, passRate, delta}}],
  aggregates{casesTotal, casesPassed, overallScore, overallPassRate, meanDelta}`. Místo `claudeVersion` pole
  `runner: {name: "skilleval", version}`, plus `models: {agent, judge}` (OpenRouter slugy) a `extensions` u case (co se použilo).
- `report.html` – jeden soběstačný soubor bez externích requestů (inline CSS/JS): verdikt a dlaždice (suite score, Δ, baseline,
  cases passing, perfect runs), karta case (Δ, skóre s tickem prahu, červený okraj při Δ<0), runy with pak without, gradery
  s chipem pass/fail, rozbalené neúspěšné s vysvětlením, u `llm` hlasy a evidence, badge `plugin-fired indicator` u nescored,
  **prohlížeč trace** (tool calls s vstupy/výstupy, zvýraznění faultů a mocků), prompt a rubriky graderů dole.
- `traces/…jsonl`, `mock-recordings/` + `ADOPT.txt`.
- `--json [path]` – totéž JSON na stdout / do souboru, bez progressu.

## 10. Testy a akceptace

- Unit testy bez sítě: loader (všechny příklady z dokumentace + chybové stavy: neznámý klíč, case bez graderu, špatný regex
  v `expect`), každý grader, scoring a ablace (`scored: false` pravidla), mock substituce/expect/replay, faults, pojmenování MCP
  toolů, expanze `${VAR}`, exit kódy, cost ceiling.
- Integrační: `FakeChatModel` se skriptovanými tool calls + lokální MCP server (FastMCP, stdio i HTTP) v `tests/fixtures/plugin/`
  → celý `skilleval run` → porovnání `aggregate-result.json` se snapshotem (bez časů/nákladů).
- Kompatibilita: spusť stejný case v `claude plugin eval` (pokud je k dispozici) a v skill-eval, porovnej strukturu JSON
  (klíče) a tvar trace řádků; `skilleval report` umí načíst i `aggregate-result.json` z CC.
- Live (opt-in `SKILLEVAL_LIVE=1`): `skilleval run skills/webstore-core --case najdi-objednavku-71781
  --runs 1 --ablation none --mocks off --model haiku` proti běžícímu MCP serveru (token v `WEBSTORE_MCP_TOKEN`) projde bez úprav sady a bez tokenu
  natvrdo v `.mcp.json`.
- `ruff check .` čistý, `pytest -q` zelený.

**Akceptační kritéria v1:**
1. Všechny cases v `skills/webstore-*/evals` projde `skilleval validate` bez chyb.
2. `skilleval run skills/webstore-core --ablation none --mocks off --runs 1 --model haiku` doběhne, vytvoří JSON + report,
   náklady odpovídají OpenRouter `usage.cost`.
3. S `--ablation with-without` je without-arm bez skillů i MCP toolů a Δ se počítá dle pravidel.
4. Case s mocky (`fixed`, `error`, `expect`, `agent`) funguje bez reálného serveru.
5. `konflikt-verze` s `skilleval.yaml` (fault conflict na 1. ostrém `orders_update`) deterministicky vyvolá conflict.

## 11. Mimo rozsah v1

Bash/WebFetch/WebSearch a OS sandbox, subagenti (`Agent`), hooky pluginu, `claude plugin eval init` s interaktivním
návrhem cases, publikace reportu, instalované pluginy z marketplace, `NotebookRead`. Návrh na v2: Bash přes `bubblewrap`,
`Agent` jako vnořený graf.

## 12. Poznámky k reálným sadám (pro autora skill-eval i skillů)

- Některé prompty obsahují v závorce popis zinscenované situace, kterou bez fault injection nelze vyvolat a kterou model
  vidí jako součást zadání: `webstore-core/konflikt-verze` („někdo změní objednávku – tool vrátí conflict“),
  `webstore-shipping/chyba-api-pri-podani` („dopravce odmítne zásilku“), `webstore-core/omezeny-uzivatel-stara-objednavka`
  (rights 5 – potřebuje `mcp_profile`). Pro ně připrav ukázkové `skilleval.yaml` (v repu sady, vlastník skillu), závorku
  z promptu ať vlastník skillu odstraní až potom.
- Zápisové cases (`zapis-bez-dotazu`, `vyridit-objednavku-71781`, `prisla-dodavka` …) mění dev DB (objednávka 71781 apod.).
  Bez `--hooks` a `setup`/`teardown` je spouštěj jen vědomě; testovací data sady se nesmí mazat. Doporučení: tag `db-write` přes `tags_extra` a v CI je vynechat.
- `expected.md` v cases je dokumentace pro lidi – skill-eval ho nečte.
