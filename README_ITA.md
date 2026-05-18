# Descrizione del problema

Abbiamo un insieme di contenitori eterogenei, ognuno dei quali è caratterizzato da:

- *type*: una stringa che descrive il tipo (identificatore univoco);
- *width*, *depth*, *height*: dimensioni del contenitore;
- *maxWeight*: il peso massimo che un singolo contenitore può sopportare;
- *cost*: il costo di utilizzo;
- *maxValue*: valore massimo che un contenitore può contenere (se non specificato, si assume *maxValue* = +∞);
- *gravityStrength*: percentuale minima della superficie di base dell'oggetto che deve essere supportata (a contatto) da superfici sottostanti (altri oggetti o il fondo del contenitore) affinché la posizione sia valida.

Gli oggetti invece sono caratterizzati da:

- *id*: per identificare il singolo oggetto;
- *width*, *depth*, *height*, *weight*: caratteristiche fisiche dell'oggetto;
- *value*: valore dell'oggetto;
- *allowedRotations*: stringa contenente i codici delle rotazioni ammesse (es. "01" indica che sono consentite le rotazioni 0 e 1); le opzioni sono:
  - 0: Nessuna rotazione;
  - 1: Rotazione attorno all'asse z di 90°;
  - 2: Rotazione attorno all'asse x di 90°;
  - 3: Rotazione attorno all'asse x di 90°, poi attorno all'asse z di 90°;
  - 4: Rotazione attorno all'asse y di 90°;
  - 5: Rotazione attorno all'asse z di 90°, poi attorno all'asse x di 90°.

Le rotazioni sono applicate rispetto agli assi del sistema di riferimento del contenitore e determinano una permutazione delle dimensioni dell'oggetto (width, depth, height).

(**N.B.**: l'asse x corrisponde alla dimensione della lunghezza (depth), l'asse y della larghezza (width) e l'asse z dell'altezza (height).)

## Obiettivo

Dato un insieme di contenitori e oggetti, determinare:

- quali contenitori utilizzare;
- come posizionare gli oggetti nei contenitori;

minimizzando il costo totale dei contenitori utilizzati, rispettando tutti i vincoli fisici e logistici.

È possibile utilizzare un numero illimitato di contenitori per ciascun tipo.  
Il costo totale è dato dalla somma dei costi dei contenitori utilizzati (indipendentemente dal loro grado di riempimento).

La soluzione deve, per ogni oggetto, definire:

- *type_vehicle*: tipo del veicolo che contiene l'oggetto;
- *idx_vehicle*: indice (consecutivo) del veicolo che contiene l'oggetto;
- *id_item*: id dell'oggetto;
- *x_origin*: minima coordinata x dell'oggetto;
- *y_origin*: minima coordinata y dell'oggetto;
- *z_origin*: minima coordinata z dell'oggetto;
- *orient*: orientamento (rotazione) dell'oggetto.

Il sistema di riferimento ha origine nell'angolo inferiore anteriore sinistro del contenitore.  
*x_origin*, *y_origin*, e *z_origin* rappresentano quindi le coordinate dell'angolo inferiore anteriore sinistro dell'oggetto.

## Vincoli

- Ogni oggetto deve essere assegnato ad esattamente un contenitore;
- Gli oggetti non possono intersecarsi: i volumi occupati devono essere disgiunti;
- Gli oggetti devono rimanere all'interno dei limiti del contenitore;
- Il peso totale degli oggetti in un contenitore non può superare *maxWeight*;
- Il valore totale non può superare *maxValue* (se definito);
- Le rotazioni devono rispettare *allowedRotations*;
- Il vincolo di stabilità deve rispettare *gravityStrength*;
- È consentito posizionare oggetti uno sopra l'altro, purché siano rispettati i vincoli di stabilità e peso.

# Esempio di dataset

## Contenitori

| type    | width | depth | height | maxWeight | cost | maxValue | gravityStrength |
|--------|------|------|--------|-----------|------|----------|----------------|
| PalletA | 1.2 | 0.8 | 1.65 | 1500 | 10 | 1000 | 0 |
| PalletB | 1.2 | 0.8 | 0.8  | 1500 | 10 |  | 50 |

## Oggetti

| id     | width | depth | height | weight | value | allowedRotations |
|--------|------|------|--------|--------|-------|------------------|
| item-1 | 1.2  | 0.8  | 0.6  | 10 | 5  | 012345 |
| item-2 | 1.2  | 0.8  | 0.6  | 10 | 20 | 0 |
| item-3 | 1.2  | 0.8  | 0.6  | 10 | 7  | 01 |
| item-4 | 0.2  | 0.5  | 0.8  | 1  | 14 | 012345 |
| item-5 | 0.26 | 0.46 | 0.81 | 10 | 1  | 014 |

# Esempio di soluzione

| type_vehicle | idx_vehicle | id_item | x_origin | y_origin | z_origin | orient |
|-------------|------------|--------|----------|----------|----------|--------|
| PalletA | 0 | item-1 | 0 | 0 | 0 | 0 |
| PalletA | 0 | item-2 | 0 | 0 | 0.8 | 0 |
| PalletB | 1 | item-3 | 0 | 0 | 0 | 0 |
| PalletB | 1 | item-4 | 0.8 | 0 | 0 | 3 |
| PalletB | 2 | item-5 | 0 | 0 | 0 | 1 |

## Regole

- **Limite di tempo**: 10 minuti  
- **Numero massimo di thread**: 4  
- **Scadenza**: 23/06/2026 23:59  
- dopo la scadenza, tutti i codici verranno eseguiti su un gruppo di 10 istanze nascoste (non disponibili tra quelle condivise).  
- **Giornata per domande e chiarimenti**: 30/06/2026  

# Richiesta

Usando python come linguaggio di programmazione e qualsiasi tipo di codice open source, sviluppare un'euristica per risolvere il problema. Successivamente, è necessario utilizzare l'euristica su tutte le istanze fornite producendo, per ciascuna, una soluzione.

Infine, è necessario caricare, nella sezione "Elaborati" del Portale della Didattica una cartella compressa contenente il codice dell'euristica e le soluzioni ottenute per tutte le istanze fornite.

## Requisiti tecnici

Il progetto deve essere svolto in gruppi di 1-4 persone.

Un solo membro del gruppo dovrà, entro la deadline, allegare una singola cartella compressa nella sezione "Elaborati" del Portale della Didattica con il nome:

`assignment_XXXXXX_YYYYYY_ZZZZZZ.zip`

dove `XXXXXX`, `YYYYYY` e `ZZZZZZ` sono i numeri di matricola dei membri del gruppo. Per esempio, se un gruppo è composto da due persone con matricole 123456 e 654321, la cartella deve essere chiamata `assignment_123456_654321.zip`, (l'ordine delle matricole è indifferente) mentre se lo studente con matricola 999999 decide di svolgere il progetto individualmente, la cartella deve essere chiamata `assignment_999999.zip`.

All'interno della cartella devono essere presenti:

- un file soluzione  
  `sol_DatasetN_solver_XXXXXX_YYYYYY_ZZZZZZ.csv`  solver_338874
  per ciascun `DatasetN`;

- il solver  
  `solver_XXXXXX_YYYYYY_ZZZZZZ.py`;

- il file `__init__.py` per poter importare il solver nel main.

- il file `requirements.txt` contenente tutte le dipendenze necessarie per l'esecuzione del solver (il file deve permettere la creazione di un ambiente virtuale funzionante tramite il comando `pip install -r requirements.txt`);

- (facoltativamente) script addizionali utilizzati dal solver.

**N.B.**: il solver deve essere eseguibile senza modifiche in un ambiente Python standard (versione 3.x), utilizzando esclusivamente le librerie specificate nel file `requirements.txt`. Inoltre è consentito utilizzare esclusivamente librerie open-source, liberamente installabili tramite `pip` e senza necessità di licenze proprietarie o commerciali.

All'interno del solver deve essere presente una classe con lo stesso nome del file che la contiene:

`class solver_XXXXXX_YYYYYY_ZZZZZZ():`

In questa classe è necessario implementare il metodo `solve()`, che non deve prendere nulla in input (i dati necessari sono presenti in `self.inst`) e non deve restituire nulla come output, ma deve occuparsi di creare il file soluzione `sol_DatasetN_solver_XXXXXX_YYYYYY_ZZZZZZ.csv` (per `DatasetN`).

(È facoltativo, ma caldamente consigliato, includere nella cartella compressa il file `abstract_solver.py` - fornito nella cartella esempio `solver_000000/` - e far sì che il proprio solver erediti dalla classe `AbstractSolver()` in esso contenuta. Questo permette di accedere al metodo `write_solution_to_file()` per creare correttamente il file soluzione.)

Un esempio di solver (`solver_000000`) è presente nel progetto. I file che devono essere inclusi nella cartella compressa sono tutti gli elementi contenuti in `solver_000000/` insieme a quelli (per ora assenti) contenuti in `results/`.

Per testare il corretto funzionamento del solver:

- in `main.py` sostituire riga 2  
  `from solver_000000 import solver_000000`  
  con  
  `from solver_XXXXXX_YYYYYY_ZZZZZZ import solver_XXXXXX_YYYYYY_ZZZZZZ`

- e riga 10  
  `solver = solver_000000(inst)`  
  con  
  `solver = solver_XXXXXX_YYYYYY_ZZZZZZ(inst)`

- eseguire `main.py`

- eseguire `results_checker.py` per verificare che la soluzione trovata dal solver sia feasible e per visualizzarne il costo totale.

**N.B.**: I solver consegnati verranno testati per la valutazione esattamente come descritto nella sezione precedente, quindi con gli stessi identici strumenti messi a disposizione durante lo sviluppo. Per questo motivo, se `main.py` e `results_checker.py`, nella versione fornita, producono un errore (a causa di nome file errato, nome della classe errato, formato di output errato o per qualsiasi altro motivo) quando vengono eseguiti, al progetto verranno automaticamente assegnati 0 punti.

# Classifica

Verrà stilata una classifica per ciascuna delle istanze fornite, in base al costo della funzione obiettivo della soluzione (feasible) fornita (costo minore = piazzamento più alto in classifica). Ogni gruppo riceverà punti pari alla posizione in classifica (prima posizione = 1 punto, seconda posizione = 2 punti, ecc.). La classifica complessiva verrà creata sommando, per ogni gruppo, i punti ottenuti su tutte le istanze e ordinando i gruppi per punteggio complessivo più basso. Le prime 10 posizioni verranno riclassificate aggiungendo i punteggi ottenuti su un nuovo insieme di istanze. Inoltre, a campione, alcune euristiche verranno testate per verificare che le soluzioni fornite siano coerenti e ripetibili.

## FAQ

- Q1 ......?
