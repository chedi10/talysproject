from faker import Faker
import pandas as pd
import numpy as np
import random
from pathlib import Path

# =========================
# CONFIG (tu peux modifier)
# =========================
SEED = 42
N_CLIENTS = 5000
AVG_CREDITS_PER_CLIENT = 3          # moyenne de crédits par client
AVG_TX_PER_CLIENT = 70              # moyenne transactions par client
DEFAULT_RATE = 0.115                # 11.5% (comme votre doc)
RAW_DIR = Path("data/raw")

fake = Faker("fr_FR")
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

RAW_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) CLIENTS
# Logique: revenu dépend de la profession (simple mais réaliste)
# =========================
PRO_JOBS = [
    ("Etudiant", 0.6, (0, 500)),
    ("Ouvrier", 1.0, (600, 1400)),
    ("Employé", 1.0, (900, 2200)),
    ("Commerçant", 1.0, (1200, 3500)),
    ("Fonctionnaire", 1.0, (1300, 3000)),
    ("Cadre", 1.0, (2500, 6000)),
    ("Indépendant", 1.0, (800, 5000)),
    ("Retraité", 1.0, (600, 2500)),
]

CITIES = ["Tunis", "Ariana", "Ben Arous", "Sousse", "Sfax", "Nabeul", "Bizerte", "Monastir", "Kairouan", "Gabès"]


def pick_job_and_salary():
    job, _, (a, b) = random.choice(PRO_JOBS)
    salary = int(np.clip(np.random.normal((a + b) / 2, (b - a) / 6), a, b))
    return job, salary


def generate_clients(n=N_CLIENTS):
    rows = []
    for cid in range(1, n + 1):
        job, salary = pick_job_and_salary()
        age = random.randint(21, 70)
        rows.append(
            {
                "client_id": cid,
                "cin": str(fake.unique.random_number(digits=8, fix_len=True)),
                "nom": fake.last_name(),
                "prenom": fake.first_name(),
                "age": age,
                "ville": random.choice(CITIES),
                "profession": job,
                "revenu_mensuel": salary,
                "date_creation": str(fake.date_between(start_date="-5y", end_date="today")),
                "statut_kyc": random.choice(["OK", "A_VERIFIER", "RISQUE"]),
            }
        )
    return pd.DataFrame(rows)


# =========================
# 2) CREDITS
# Logique: montant dépend du revenu + DTI; défaut ~ 11.5%
# =========================
CYCLES = ["CYCLE_1", "CYCLE_2", "CYCLE_3", "CYCLE_4"]
PURPOSES = ["CONSOMMATION", "MICRO_ENTREPRISE", "SANTE", "EDUCATION", "LOGEMENT"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_credits(df_clients, avg_per_client=AVG_CREDITS_PER_CLIENT, default_rate=DEFAULT_RATE):
    rows = []
    credit_id = 1

    for _, c in df_clients.iterrows():
        k = max(1, int(np.random.poisson(avg_per_client)))
        salary = c["revenu_mensuel"]

        for _ in range(k):
            cycle = random.choices(CYCLES, weights=[0.45, 0.30, 0.18, 0.07])[0]

            # montant dépend du revenu et du cycle
            cycle_mult = {"CYCLE_1": 1.0, "CYCLE_2": 1.4, "CYCLE_3": 1.8, "CYCLE_4": 2.2}[cycle]
            base_amount = np.clip(np.random.normal(loc=salary * cycle_mult * 2.2, scale=salary * 0.8), 300, 25000)
            amount = int(base_amount)

            # dette/revenu (DTI) simulé (0.05 -> 1.2)
            dti = float(np.clip(np.random.beta(2, 6) * 1.4, 0.05, 1.2))

            # prob défaut dépend du dti + statut kyc + revenu
            kyc_penalty = {"OK": 0.0, "A_VERIFIER": 0.5, "RISQUE": 1.0}[c["statut_kyc"]]
            income_factor = -0.00025 * salary  # revenu ↑ => défaut ↓
            logit = 1.5 * (dti - 0.55) + kyc_penalty + income_factor
            p_default = float(sigmoid(logit))

            # calibration simple vers ~11.5% global
            p_default = float(np.clip(p_default * (default_rate / 0.20), 0.01, 0.95))

            defaulted = 1 if random.random() < p_default else 0

            start_date = fake.date_between(start_date="-3y", end_date="today")
            duration_months = random.choice([6, 9, 12, 18, 24])

            rows.append(
                {
                    "credit_id": credit_id,
                    "client_id": int(c["client_id"]),
                    "cycle": cycle,
                    "objet": random.choice(PURPOSES),
                    "montant": amount,
                    "duree_mois": duration_months,
                    "dti": round(dti, 3),
                    "date_debut": str(start_date),
                    "en_defaut": defaulted,
                }
            )
            credit_id += 1

    return pd.DataFrame(rows)


# =========================
# 3) REMBOURSEMENTS
# Logique: bons payeurs -> 80% à temps (retard moyen faible)
#         défaut -> retards croissants (retard moyen élevé)
# =========================
def generate_repayments(df_credits):
    rows = []
    pay_id = 1

    for _, cr in df_credits.iterrows():
        dur = int(cr["duree_mois"])
        amount = float(cr["montant"])
        monthly = round(amount / dur, 2)

        defaulted = int(cr["en_defaut"])

        for m in range(1, dur + 1):
            # date échéance = date_debut + m mois (approx)
            due = pd.to_datetime(cr["date_debut"]) + pd.DateOffset(months=m)

            if defaulted == 0:
                # 80% à temps, sinon petit retard
                on_time = random.random() < 0.80
                delay_days = 0 if on_time else int(np.clip(np.random.normal(2, 3), 1, 15))
            else:
                # retards qui augmentent
                base = 15 + (m * 2)
                delay_days = int(np.clip(np.random.normal(base, 15), 5, 120))

            paid_date = due + pd.Timedelta(days=delay_days)
            statut = "PAYE" if delay_days < 90 else "EN_RETARD"

            rows.append(
                {
                    "remb_id": pay_id,
                    "credit_id": int(cr["credit_id"]),
                    "client_id": int(cr["client_id"]),
                    "mois": m,
                    "montant_du": monthly,
                    "date_echeance": str(due.date()),
                    "date_paiement": str(paid_date.date()),
                    "retard_jours": delay_days,
                    "statut": statut,
                }
            )
            pay_id += 1

    return pd.DataFrame(rows)


# =========================
# 4) TRANSACTIONS
# Logique: 35% dépôts, 30% retraits, 20% remboursements, 15% transferts
# =========================
TX_TYPES = ["DEPOT", "RETRAIT", "REMBOURSEMENT", "TRANSFERT"]
TX_WEIGHTS = [0.35, 0.30, 0.20, 0.15]


def generate_transactions(df_clients, avg_per_client=AVG_TX_PER_CLIENT):
    rows = []
    tx_id = 1

    for _, c in df_clients.iterrows():
        n = max(5, int(np.random.poisson(avg_per_client)))
        salary = c["revenu_mensuel"]

        for _ in range(n):
            ttype = random.choices(TX_TYPES, weights=TX_WEIGHTS)[0]
            # montant adapté au revenu
            if ttype == "DEPOT":
                amount = np.clip(np.random.normal(salary * 0.4, salary * 0.25), 10, 20000)
            elif ttype == "RETRAIT":
                amount = np.clip(np.random.normal(salary * 0.25, salary * 0.20), 5, 15000)
            elif ttype == "REMBOURSEMENT":
                amount = np.clip(np.random.normal(salary * 0.18, salary * 0.12), 10, 12000)
            else:  # TRANSFERT
                amount = np.clip(np.random.normal(salary * 0.22, salary * 0.35), 1, 30000)

            date = fake.date_between(start_date="-2y", end_date="today")

            # simple "suspect" si transferts très grands
            suspect = 1 if (ttype == "TRANSFERT" and amount > (salary * 3)) else 0

            rows.append(
                {
                    "transaction_id": tx_id,
                    "client_id": int(c["client_id"]),
                    "type": ttype,
                    "montant": round(float(amount), 2),
                    "date": str(date),
                    "suspect": suspect,
                }
            )
            tx_id += 1

    return pd.DataFrame(rows)


# =========================
# 5) RELATIONS (Graph)
# Logique: garant d’un client en défaut -> "contagion" de risque
# =========================
REL_TYPES = ["GARANT", "FAMILLE", "BUSINESS"]


def generate_relations(df_clients, df_credits, n_edges=18000):
    # clients en défaut
    default_clients = set(df_credits[df_credits["en_defaut"] == 1]["client_id"].unique())

    rows = []
    rel_id = 1
    client_ids = df_clients["client_id"].tolist()

    for _ in range(n_edges):
        src = random.choice(client_ids)
        tgt = random.choice(client_ids)
        while tgt == src:
            tgt = random.choice(client_ids)

        rtype = random.choices(REL_TYPES, weights=[0.35, 0.40, 0.25])[0]

        # contagion: si relation GARANT vers un client en défaut => risque relation élevé
        if rtype == "GARANT" and tgt in default_clients:
            risk = random.randint(70, 100)
        else:
            risk = random.randint(5, 80)

        rows.append(
            {
                "relation_id": rel_id,
                "source_client_id": int(src),
                "target_client_id": int(tgt),
                "type_relation": rtype,
                "risk_relation": risk,
            }
        )
        rel_id += 1

    return pd.DataFrame(rows)


# =========================
# MAIN
# =========================
def main():
    print("1) Génération clients...")
    df_clients = generate_clients()
    df_clients.to_csv(RAW_DIR / "clients.csv", index=False)
    print("   -> data/raw/clients.csv OK")

    print("2) Génération credits...")
    df_credits = generate_credits(df_clients)
    df_credits.to_csv(RAW_DIR / "credits.csv", index=False)
    print("   -> data/raw/credits.csv OK")

    print("3) Génération remboursements...")
    df_remb = generate_repayments(df_credits)
    df_remb.to_csv(RAW_DIR / "remboursements.csv", index=False)
    print("   -> data/raw/remboursements.csv OK")

    print("4) Génération transactions...")
    df_tx = generate_transactions(df_clients)
    df_tx.to_csv(RAW_DIR / "transactions.csv", index=False)
    print("   -> data/raw/transactions.csv OK")

    print("5) Génération relations...")
    df_rel = generate_relations(df_clients, df_credits)
    df_rel.to_csv(RAW_DIR / "relations.csv", index=False)
    print("   -> data/raw/relations.csv OK")

    print("\nTerminé ! Fichiers générés dans data/raw/")
    print("   - clients.csv")
    print("   - credits.csv")
    print("   - remboursements.csv")
    print("   - transactions.csv")
    print("   - relations.csv")


if __name__ == "__main__":
    main()

