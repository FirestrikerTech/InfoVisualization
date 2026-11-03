# Interaktive 3D Loss-Landschaft Live-Demo (MNIST)

Live-Demo für den Vortrag über die Visualisierung von Loss-Landschaften nach **Li et al. (NeurIPS 2018), *„Visualizing the Loss Landscape of Neural Nets“***.

- **Live-URL (GitHub Pages):** https://firestrikertech.github.io/InfoVisualization/
- **QR-Code für Zuhörer:** [`docs/qr.png`](docs/qr.png)
- **Direktlink für den Vortragenden (schaltet sofort alle Phasen + Filternormierung frei):**
  https://firestrikertech.github.io/InfoVisualization/#presenter

## Ablauf im Vortrag (3 Phasen)

1. **Phase 1 – Manuelle Richtungserkundung (`Richtungs-Mixer`):**
   - Die Zuhörer stehen fest im trainierten Minimum $\theta^*$ ($\alpha=0, \beta=0$) und mischen intuitive Grundrichtungen im 25.450-dimensionalen Parameterraum (`784 → 32 → 10` ReLU-MLP).
   - Man erkennt: Mit niedrigdimensionalen/strukturierten Richtungen sieht man meist nur flache Rinnen oder schiefe Ebenen. Über den Button *„Tipp: Schichten koppeln“* (oder geduldiges Mischen von Schicht 1 + Schicht 2 + Drehwinkel) lässt sich ein lokales Tal erahnen.
2. **Phase 2 – Gaußsche Zufallsrichtungen ohne Filternormierung (nach 2,5 Minuten):**
   - Nach 150 Sekunden (einstellbar in `docs/assets/meta.json` unter `unlock_random_seconds`) erscheint animiert der Button **„🎲 Gauß-Zufallsrichtungen erzeugen“** ($\delta, \eta \sim \mathcal{N}(0, I)$).
   - **Der Skalierungs-Trick:** Da ReLU positiv homogen ist, wurden die Gewichte beim Export mit $c = 10$ skaliert ($W_1, b_1 \leftarrow c \cdot W_1, c \cdot b_1$ und $W_2 \leftarrow W_2 / c$). Das Netz berechnet **exakt dieselbe Funktion** (identischer Loss `0.128`, identische Test-Accuracy `95.8%`), aber unnormierte Zufallsrichtungen stören $W_2$ relativ gesehen 100× stärker → es entsteht ein **extrem spitzer Trichter**.
3. **Phase 3 – Versteckte Filternormierung (Li et al., 2018):**
   - **Geheime Geste auf dem Handy:** Halte den Finger **3 Sekunden lang** oben links auf das Feld **`Loss L(θ*)`** gedrückt (oder öffne die URL mit `#presenter`).
   - Dadurch erscheint die grüne Box **„✨ Filternormierung (Li et al., 2018)“**, die jede Neuronen-Zeile auf die Norm des jeweiligen Filters normiert ($d_{i,j} \leftarrow \frac{d_{i,j}}{\lVert d_{i,j}\rVert}\lVert\theta_{i,j}\rVert$, Bias $= 0$).
   - Sofort verwandelt sich der spitze Trichter in das **flache, breite, generalisierende Tal**.

## Lokal starten / testen

```bash
python3 -m http.server 8000 --directory docs
```

Danach im Browser öffnen:
- Zuhörer-Ansicht: http://localhost:8000/
- Vortragenden-Ansicht: http://localhost:8000/#presenter
