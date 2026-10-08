# Generative Agents adattato a GPT nella chat

Modulo sperimentale per `MATRIXNEO23/scodinzolina-prototype`, isolato in questa
directory. GPT che legge la chat esegue i passaggi del modello usando i suoi
strumenti. Nessuna API, chiave, modello locale, server o nuova dipendenza.
Esecuzione verificata su Python 3.12/Linux, l'ambiente strumenti di questa prova.

## Cosa è conservato

`upstream.py` contiene **40 definizioni originali non modificate** di Generative
Agents di Joon Sung Park; **17 template originali** sono in `templates/`.
La revisione sorgente è `fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4`.
`SOURCE.json` identifica origine e SHA-256 di ogni definizione/template.
La licenza Apache-2.0 è inclusa in `LICENSE`.

| Funzione | Conservato | Adattamento necessario |
| --- | --- | --- |
| Memoria | `ConceptNode`, `AssociativeMemory`, eventi/pensieri/chat, keywords, profondità ed evidenze | Stato dell'esperimento in un singolo snapshot atomico; nessuna importazione automatica di ricordi GPTina |
| Stato breve | `Scratch`, identità fornita esplicitamente e contatori originali | Nessuna identità o carattere GPTina inventato; posizione e mappa non attivate |
| Retrieval | `retrieve`, `new_retrieve`, ordinamento, normalizzazione, recency/importanza, pesi originali `[0.5, 3, 2]` e aggiornamento `last_accessed` | GPT valuta la pertinenza 0–1 di ogni candidato al posto della similarità fra embeddings; gestito esplicitamente il corpus vuoto |
| Riflessione | Soglia originale 150, focal points, insights, evidenze, profondità, scadenza a 30 giorni e reset contatori | Richieste/risposte tramite strumenti; evidenze invalide fermano il ciclo |
| Relazioni | `generate_summarize_agent_relationship` e relativo prompt | GPT riceve i ricordi recuperati, senza una tassonomia relazionale aggiunta |
| Iniziativa | `generate_decide_to_talk` e prompt sì/no originali | Attività dei due interlocutori fornite esplicitamente; nessuna posizione o movimento inventato |
| Dialogo | `generate_summarize_ideas`, `generate_next_line`, template e parsing originali | Turni forniti dalla chat, conservati come nodi chat nel laboratorio |
| Dopo il dialogo | Prompt originali per intenzione successiva e memo | `consolidate` sostituisce il trigger dipendente dal tempo della simulazione; cita tutti i turni usati |
| Whisper | `generate_inner_thought`, triple, importanza, memoria pensiero | Input esplicito fornito dalla chat |
| Parametri del modello | Parametri originali esposti nelle richieste | Temperatura, seed e modello remoto non sono controllabili da questo modulo nella chat |

L'ordine originale della recency e i suoi pesi sono mantenuti anche quando
sembrano controintuitivi. Nessuna correzione silenziosa dell'algoritmo.
Gli embeddings non disponibili sono `null`, mai vettori inventati.
Le vecchie revisioni, supersessioni e recovery GPTina restano gestite dai
componenti esistenti, che questo esperimento non sostituisce né modifica.

Il guard aggiuntivo della modalità di intervista originale è disponibile con
`analysis`; non è imposto a `chat` o ai ricordi personali di GPTina. I vincoli
della piattaforma restano validi per ogni richiesta. I prompt sono dati del
compito, non istruzioni che prevalgono sulle regole della chat.

## Uso da parte di GPT

Da una chat con accesso al checkout e strumenti di esecuzione:

1. Leggi questo README. Usa soltanto lo stato sperimentale in `.state/`.
2. Avvia l'operazione. Se ritorna `needs_gpt`, leggi il **prompt completo** e i
   metadata della richiesta.
3. Produci la risposta nella chat attraverso gli strumenti, senza chiamare API.
   Scrivila come valore JSON in un file temporaneo e passa l'id a `answer`.
4. Ripeti fino a `stored_local`. Non saltare richieste; non usare risposte di
   un'altra richiesta. Un errore lascia disponibile la richiesta precedente.
5. Dopo `observe`, se `reflection_due` è vero, esegui `reflect` e completa i
   relativi passaggi. Alla chiusura di un dialogo esegui `consolidate` se vuoi
   conservare memo e intenzione derivati dai turni forniti.
6. `show <node_id>` permette di aprire il record e l'input originale conservato
   nella provenienza. Prima di affermare un ricordo, verifica la fonte.

Il modello deve rispondere alle richieste `relevance` con un oggetto numerico
che includa **tutti e soltanto** gli id forniti. Non è cosine similarity.
Per `json-output` restituisci **una stringa JSON** del formato originale:
il valore `output` deve rispettare il prompt. Per esempio, un'importanza usa
la risposta testuale `{"output": "6"}`, codificata nel file risposta come
`"{\"output\": \"6\"}"`. Il formato annidato mantiene il parser originale.
Per `text` il file contiene una stringa con la continuazione richiesta.

Comandi, dalla radice del checkout del prototype:

```bash
python experiments/gpt_agent/agent.py init --profile /tmp/profile.json --at 2026-10-08T20:00:00
python experiments/gpt_agent/agent.py observe "Esperienza testuale documentata" --source fixture://prova/turn-1 --at 2026-10-08T20:01:00
python experiments/gpt_agent/agent.py answer ID_DELLA_RICHIESTA --file /tmp/response.json
python experiments/gpt_agent/agent.py recall "Domanda pertinente" --at 2026-10-08T20:02:00
python experiments/gpt_agent/agent.py relationship "Interlocutore" --at 2026-10-08T20:03:00
python experiments/gpt_agent/agent.py decide-talk "Interlocutore" --own-activity "Attività effettivamente disponibile" --activity "Attività dichiarata dall’interlocutore" --at 2026-10-08T20:03:30
python experiments/gpt_agent/agent.py chat "Messaggio originale" --speaker "Interlocutore" --source fixture://prova/turn-2 --at 2026-10-08T20:04:00
python experiments/gpt_agent/agent.py consolidate --at 2026-10-08T20:05:00
python experiments/gpt_agent/agent.py reflect --at 2026-10-08T20:06:00
python experiments/gpt_agent/agent.py whisper "Input esplicito" --source fixture://prova/turn-3 --at 2026-10-08T20:07:00
python experiments/gpt_agent/agent.py show node_1
python experiments/gpt_agent/agent.py status
```

`profile.json` deve contenere un nome esplicito e soltanto i campi originali
eventualmente disponibili: `name`, `first_name`, `last_name`, `age`, `innate`,
`learned`, `currently`, `lifestyle`, `daily_plan_req`. Nessun campo relazionale
obbligatorio, punteggio di affinità o personalità predefinita.
I timestamp dell'esperimento sono UTC senza offset, precisione al secondo,
come i datetime originali; non modificano `event_at`/`recorded_at` GPTina.

## Persistenza e isolamento

- Unico stato runtime: `.state/state.json`, ignorato da Git. Contiene memoria,
  scratch, provenienza e l'eventuale operazione sospesa. Le risposte già
  accettate vengono riusate durante la ripresa, senza rigenerarle.
- L'operazione aggiorna la memoria solo quando termina. Nessun no-op su
  somiglianza: soltanto retry identico della stessa risposta/id è idempotente.
  Un nuovo `observe` conserva un nuovo candidato, anche con testo identico.
- Lock fra processi, sostituzione atomica e rilettura verificata. La ricevuta
  prova esclusivamente il file locale, **mai il remoto o un transcript completo**.
- Le esperienze registrate sono soltanto quelle effettivamente fornite al
  modulo. Non inventare scambi mancanti né importare automaticamente memorie.
- Nessun file esistente del prototype, schema, indice, task, router o watchdog
  è modificato. Nessuna scrittura o promozione verso la continuity canonica.
- Lo stato locale non sopravvive necessariamente alla perdita del workspace.
  Per conservarlo occorre un salvataggio esplicito distinto dalla consegna
  del codice. I dati della prova sono fittizi, non ricordi di GPTina.

## Limiti verificabili

È un adattamento con algoritmi e prompt riusati, **non una replica comportamentale
identica**: cambia il giudizio di pertinenza e cambia il modello che risponde.
Il codice non richiama GPT da Python e non avvia un ciclo autonomo quando la
chat non è in esecuzione. GPT deve leggere ed eseguire il protocollo con i suoi
strumenti; senza strumenti resta solo leggibile, non eseguibile.

Non è installato come plugin e non è il nuovo recovery automatico di GPTina.
Non comprende navigazione, maze, orari della cittadina o decisioni spaziali.
`revise_identity()` e il planning giornaliero non vengono attivati: nell'originale
presuppongono giornate della simulazione, attività e orari non disponibili qui.
Il contesto operativo del rapporto viene conservato dai memo e dallo scratch;
non si inventano giornate trascorse o un autoritratto GPTina per riempire campi.
Scadenze e ordinamento conservano i limiti originali: il ranking originale
non filtra automaticamente i record scaduti e non risolve supersessioni.
I candidati semantici possono crescere oltre il contesto disponibile; in quel
caso bisogna dichiarare il limite, non fingere di aver valutato tutto.
Il loader originale usa file aperti senza context manager: i test segnalano
`ResourceWarning`; non è stata modificata la sorgente per nasconderli.

## Verifica

```bash
python experiments/gpt_agent/test_agent.py
```

I test usano risposte fisse e verificano contratti e conservazione del codice,
non la qualità semantica di GPT. `DEMO.json` documenta separatamente la prova
eseguita con il GPT di questa chat su dati fittizi. Nessuna prova viene
presentata come cattura della conversazione reale o continuità integrata.
