# Interaktive 3D Loss-Landschaft Live-Demo (MNIST)

Live-Demo für den Vortrag über die Visualisierung von Loss-Landschaften nach **Li et al. (NeurIPS 2018), *„Visualizing the Loss Landscape of Neural Nets“***.

- **Live-URL (GitHub Pages):** https://firestrikertech.github.io/InfoVisualization/
- **QR-Code für Zuhörer:** [`docs/qr.png`](docs/qr.png)
- **Direktlink für den Vortragenden (schaltet sofort alle Phasen + Filternormierung frei):**
  https://firestrikertech.github.io/InfoVisualization/#presenter

## Ablauf im Vortrag (3 Phasen)

1. **Phase 1 – Manuelle Richtungserkundung (4 Scroller):**
   - Die Zuhörer stehen fest im Punkt $\theta^*$ ($\alpha=0, \beta=0$) und haben genau **4 Regler**:
     - Jeweils einen Regler für Achse $\alpha$ und Achse $\beta$, um durch die Dimensionen des Parameterraums zu scrollen (`Dim #1` bis `Dim #50`), und jeweils einen Regler für die Stärke in dieser Dimension.
   - **Startzustand (kein Minimum sichtbar):** Auf `Dim #1` und `Dim #2` (Randpixel / einzelne Koordinaten) ist die Landschaft eine flache Ebene mit leichter Kante – das Minimum ist überhaupt nicht erkennbar! Erst wer beide Achsen in die gekoppelten Schichten (`Dim #43` bis `Dim #50`) scrollt und die Stärke anpasst, kann manuell eine Mulde finden.
   - **Performance (50 Bilder + Int16-Quantisierung):** Die Gewichte sind auf 16-Bit-Integer (`weights_q16.bin`, nur 50 KB) quantisiert und der Loss wird live über 50 balancierte MNIST-Bilder (`data.bin`, 39 KB) ausgewertet – ein komplettes 41×41-Gitter dauert auf dem Smartphone nur wenige Millisekunden!
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
