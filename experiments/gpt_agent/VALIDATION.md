# Verifica del modulo GPT nella chat — 8 ottobre 2026

Laboratorio: `MATRIXNEO23/scodinzolina-prototype`, base `main`
`14fba536d2ad69d8a3e4fd561dffab0bea7645e2`, verificata live.
Branch locale: `lab/gpt-chat-agent-2026-10-08`.
Commit del codice verificato: `d95145c` (preceduto da `554953f`).
Questa relazione è aggiunta dopo la verifica, senza modifiche al codice.

## Diff funzionale

Tutte le aggiunte sono sotto `experiments/gpt_agent/`. **Nessun file esistente
è modificato.** Non è una sostituzione attivata del recovery GPTina.

- `agent.py`: un solo adapter runtime (483 righe), standard library; richiede
  esplicitamente GPT, accetta la risposta dagli strumenti e riprende.
- `upstream.py`: 40 definizioni originali conservate, 2.182 righe riusate.
- `templates/`: 17 template originali.
- `SOURCE.json`, `LICENSE`: revisione, attribution e integrità verificabile.
- `test_agent.py`: test contrattuali con risposte fisse, separati dalla prova GPT.
- `DEMO.json`: prova controllata con risposte prodotte dal GPT di questa chat.
- `README.md`: comandi, protocollo per GPT, differenze e limiti.
- `.gitignore`: stato sperimentale e cache esclusi da Git.

Nessun modello locale, client API, servizio, plugin installato, daemon,
database o modifica alle automazioni. Due file Python a runtime, di cui uno
contiene il codice originale; nessun secondo engine inserito nel retrieval
GPTina. La pertinenza semantica dell'esperimento è esplicitamente diversa dalla
cosine similarity originale, mentre il calcolo restante è conservato.

## Test del modulo

`python -W ignore::ResourceWarning experiments/gpt_agent/test_agent.py`

**17/17 passati.** Coprono:

- hash delle 40 definizioni e dei 17 template originali;
- assenza di import client di rete e dei writer GPTina esistenti;
- corpus vuoto, normalizzazione/ranking e recupero `last_accessed` dopo reload;
- capture originale con fonte esplicita, senza scarto di osservazioni uguali;
- richieste sospese, retry identico, retry conflittuale, richiesta non pertinente;
- importanza/relevance valide e rifiuto di id inventati, NaN o valori booleani;
- reflection, evidenze realmente recuperate, reset della soglia originale;
- dialogo, relazione, post-conversation memo e intenzione con evidenze;
- decisione sì/no di iniziativa sociale originale;
- guard della ricerca disponibile soltanto nel modo `analysis`;
- crash simulato prima di `os.replace` e retry successivo;
- due processi che inviano la stessa risposta finale senza duplicare il record;
- ricevuta locale riletta e hashed, nessuna dichiarazione di verifica remota;
- inizializzazione che non sovrascrive, operazione occupata e precisione temporale.

Le risposte fisse provano meccanica e contratti, **non qualità del modello**.
Il loader originale genera ResourceWarning per file aperti senza context manager.
Il warning è annotato e la sorgente resta identica; il flag del comando rende
leggibile l'output dei test. Il primo controllo ha rilevato una differenza
tuple/list nella rilettura JSON: corretta con confronto canonico del contenuto.
Nessun errore residuo nei test del modulo.

## Prova reale con GPT nella chat

`DEMO.json` contiene **17 richieste e risposte** prodotte in questa istanza,
senza API né stub del modello. I fatti sono una fixture fittizia su Marco e
un Agente di prova, esplicitamente distinta da GPTina e da Alberto.

Risultato: due esperienze conservate; reload fra i comandi; retrieval che
recupera per prima la precisazione pertinente; dialogo che chiede l'obiettivo
invece di presumerlo; conversazione originale conservata; due pensieri
derivati con evidenza `node_3`; sintesi relazionale limitata alle fonti.
Stato finale: cinque nodi, nessuna richiesta sospesa.

La pipeline reflection e l'iniziativa sono coperte dai test a risposte fisse,
non da questa smoke prova GPT. Nessun test qui dimostra equivalenza perfetta,
affidabilità su larga scala, cattura dell'intera chat o integrazione approvata.

## Regressioni del prototype esistente

| Comando | Esito |
| --- | --- |
| `python rag/test_live_context.py` | PASS, compatibilità v1 e round-trip v2 |
| `python rag/test_memory_deduplication.py` | PASS, soglia/comportamento originali invariati |
| `python rag/test_memory_schema.py` | PASS |
| `python rag/test_recover_context.py` | PASS, tre casi causali/personali |
| `python rag/test_checkpoint_watchdog.py` | PASS |
| `python rag/test_source_role_audit.py` | PASS |
| `python rag/live_context.py verify` | PASS, 146 micro, 50 legacy v1 e 96 v2 |
| `python rag/checkpoint_watchdog.py` | `healthy` secondo il watchdog esistente |
| `python rag/gptina_memory.py verify` | PASS |
| `python rag/gptina_memory.py build` | PASS, ricostruzione JSONL/SQLite |
| `python rag/test_cold_start_recovery.py` | PASS, sei casi mirati, include runbook |
| `python rag/test_memory_retrieval.py` | PASS, 21 casi |
| `python rag/test_projection_resilience.py` | PASS, ricostruzione, preservazione, writer concorrenti |

Un primo `test_recover_context.py` prima del commit ha rifiutato correttamente
il worktree dirty. Rieseguito sul candidato committato: PASS. Non si è
disabilitato il controllo e non si è presentata la preview come memoria pubblicata.

Verifica aggiuntiva: il source discovery di `gptina_memory.py` non include
alcun file di `experiments/gpt_agent/`. `git diff --name-only main` fuori da
questa directory è vuoto. Nessuna migrazione o riscrittura legacy.

Questi sono i gate **già esistenti**: non provano che i difetti emersi nel
precedente audit siano corretti. In particolare il near-match del writer
GPTina rimane invariato; il watchdog esistente non dimostra copertura completa.
Questo modulo non incorpora o ricostruisce la precedente patch locale congelata.

`git diff --check` passa sui file nuovi scritti per l'adapter. Il controllo
globale segnala whitespace ereditato nei blocchi/template copiati alla lettera;
non è stato normalizzato per conservare gli hash sorgente originali.

## Isolamento e rischi residui

Nessun push. `main` remoto del prototype è ancora la base verificata.
La continuity canonica non è stata modificata da questo lavoro. Il suo HEAD
remoto letto durante la verifica è `c30398089618f18367d178cd538c7441c9a45adc`;
era avanzato rispetto alle letture precedenti, senza operazioni di scrittura
di questo incarico. Nel vecchio checkout `canonical-audit` era presente una
PNG modificata, con mtime precedente all'incarico; lasciata intatta.

Non si può promettere comportamento identico cambiando provider, né controllare
parametri del GPT della chat tramite questi file. La pertinenza è un giudizio
del modello e può variare. I candidati possono eccedere il contesto disponibile.
Nessuna inferenza viene trasformata automaticamente in memoria canonica GPTina.
L'esperimento conserva anche limiti originali, come ranking senza filtro
automatico delle scadenze e assenza di resolver delle supersessioni.

Lo stato `.state/` è locale: ricevuta e hash non sono una pubblicazione remota.
Il modulo richiede una chat con strumenti; non esegue richieste al modello da
Python e non continua da solo a chat inattiva. Non è attivato nel recovery.
Per tornare indietro basta tornare a `main`: nessun file legacy è da ripristinare.

È una **candidata sperimentale eseguibile**, non una continuity integrata e
non una sostituzione approvata del sistema GPTina.
