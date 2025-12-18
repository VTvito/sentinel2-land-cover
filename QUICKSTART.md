# ⚡ Quick Start (2 minuti)

## 1. Installa
\\ash
pip install -e .
\
## 2. Prova il demo
\\ash
python scripts/analyze_city.py --demo
\
## 3. Guarda il risultato
\\ash
# Windows
start data/demo/milan_sample/consensus.png

# Mac/Linux  
open data/demo/milan_sample/consensus.png
\
---

## Analizza una nuova città

### Opzione A: Con credenziali Copernicus (consigliato)
\\ash
# 1. Setup iniziale (una volta sola)
python scripts/setup.py

# 2. Analizza
python scripts/analyze_city.py --city Rome --download
\
### Opzione B: Download manuale
1. Scarica da [Copernicus Browser](https://browser.dataspace.copernicus.eu)
2. Estrai le bande:
   \\ash
   python scripts/extract_all_bands.py your_download.zip data/cities/rome/bands
   \3. Analizza:
   \\ash
   python scripts/analyze_city.py --city Rome
   \
---

## Comandi utili

| Comando | Descrizione |
|---------|-------------|
| \--demo\ | Usa dati di esempio (nessun download) |
| \--method kmeans\ | Classificazione veloce |
| \--method consensus\ | Classificazione accurata (default) |
| \--radius 20\ | Raggio area in km |
| \-v\ | Output verboso con progress bar |

---

📚 **Documentazione completa:** [README.md](README.md)
