# 🧠 Association Rule Mining System — NYC Open Data

This project implements a custom association rule mining system to uncover meaningful patterns across two New York City civic datasets: the NYC Rodent Inspection data (Department of Health) and OATH Administrative Hearings data (Office of Administrative Trials and Hearings). Using a modified Apriori algorithm and domain-specific filters, the system extracts non-trivial insights about how public health violations (e.g., rodent infestations) relate to administrative legal outcomes.

Built as a solo final project for an Advanced Database Systems course at Columbia University.

---

## 📊 Motivation

New York City is infamous for its rodent problem. This project investigates the bureaucratic consequences of rodent activity by analyzing whether reported rodent violations in one department (Department of Health) lead to penalties or hearings in another (OATH). Through carefully joined datasets and optimized rule mining, we aim to surface behavioral and procedural patterns within civic infrastructure.

---

## 🗃️ Datasets Used

1. **NYC OATH Hearings Division Case Status Dataset**  
   - Source: NYC Open Data  
   - Logs public safety & quality of life violation hearings.

2. **NYC Rodent Inspections Dataset**  
   - Source: NYC Department of Health & Mental Hygiene  
   - Records rat sightings, inspections, and sanitation outcomes.

After extensive filtering, datasets were joined on **ZIP code** and **violation/inspection date** to create an integrated dataset linking inspection events with administrative responses (~100K rows).

---

## 🧮 Key Features & Technical Highlights

- **Custom Apriori Implementation**
  - Built from scratch in Python based on Agrawal & Srikant (1994)
  - SQL-inspired candidate generation (Apriori-Gen) and pruning logic
  - Each item encoded as `column=value` for transparency and rule clarity

- **Domain-Specific Optimizations**
  - Redundancy filtering (e.g., eliminating `A => B` and `B => A`)
  - Semantic deduplication across location-based attributes
  - Lift-based statistical thresholding (default `min_lift=2.0`) to suppress trivial rules

- **Custom Interestingness Ranking**
  - Combined rule ranking based on:  
    - Support  
    - Confidence  
    - Lift  
    - Cross-dataset presence (Rodent + OATH attributes)  
    - Geospatial significance (ZIP code, borough)

- **Cloud Deployment**
  - Designed for scalability with full runs conducted on Google Cloud VM
  - Hyperparameter tuning (min_sup, min_conf, min_lift) via CLI inputs

---

## 🛠️ Technologies Used

- **Language:** Python 3
- **Libraries:** Pandas, NumPy
- **Environment:** Visual Studio Code (local), Google Cloud VM (deployment)
- **Input Format:** CSV datasets
- **Output:** Human-readable `.txt` file with rules ranked by interestingness

---

## 💡 Example Insights

A few standout rules from our final output:

- `[hearing status=new issuance, result=rat activity] => [compliance status=penalty due]`  
  Conf: 100% | Lift: 1.29  
  → Rodent-related violations frequently escalate into formal penalties.

- `[borough=brooklyn, hearing status=paid in full] => [compliance status=all terms met]`  
  Conf: 100% | Lift: 4.45  
  → In Brooklyn, financial resolution often correlates with broader compliance.

- `[hearing result=dismissed, result=rat activity] => [compliance status=all terms met]`  
  Conf: 99.8% | Lift: 11.99  
  → Suggests surprising remediation outcomes even when rat activity is confirmed.

---

## 🚀 Running the Code

To execute the association rule mining system:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nyc-association-rules.git
    cd nyc-association-rules
    ```

2. Ensure dependencies are installed:
    ```bash
    pip install pandas numpy
    ```

3. Place cleaned and pre-joined dataset as `INTEGRATED-DATASET.csv` in root directory.

4. Run the program:
    ```bash
    python main.py --min_sup 0.05 --min_conf 0.7 --min_lift 2.0
    ```

5. Output will be saved to `output.txt` with rules and ranked interestingness section.

---

## 📌 Notes

- This project was completed as a solo final project at Columbia University.
- All code is original and built from scratch — no off-the-shelf ML libraries used.
- Project demonstrates database theory (Apriori), systems design, and civic data analysis.

---

## 📬 Contact

Built by **[Your Name]**  
For questions or collaboration, please reach out via [your.email@example.com] or visit [your GitHub profile].

