"""sql_query — target task B: multi-turn text-to-SQL over a synthetic SQLite DB
(tiered difficulty, 3 tables).

Schema names are drawn from synonym pools per seed, so the agent must inspect
the schema and frequently hits `no such column/table` errors — the same
"read error, react" behavior that pyfix (task C) drills, in another domain.
Order revenue is unit_price × qty (derived across a join), which tier-1/2
questions need. Gold answer = result of a hidden gold query.
"""
from __future__ import annotations

import random
import sqlite3
from ..envs import TaskInstance

SYSTEM = """You are a data agent working with a SQLite database. Answer the question by querying the DB.

Tools (reply with ONE fenced action block):
```action
{"tool": "sql", "args": {"query": "SELECT ..."}}
```
executes SQL and returns up to 20 rows (or the SQLite error).
```action
{"tool": "submit", "args": {"answer": <number or string>}}
```
submits the final answer: a bare number (no units, no words) or the exact name string. The schema is NOT given: discover it first (e.g. SELECT name, sql FROM sqlite_master). You have a small turn budget, so be efficient: explore, query, submit."""

FIRST = ["Ava", "Ben", "Chen", "Dana", "Eli", "Fay", "Gus", "Hana", "Ivan", "Jia",
         "Kai", "Lena", "Mo", "Nina", "Omar", "Pia", "Quinn", "Rui", "Sam", "Tia"]
CITY = ["Austin", "Boston", "Chicago", "Denver", "Elko", "Fresno", "Geneva", "Hilo"]
PRODN = ["laptop", "phone", "tablet", "monitor", "camera", "router", "drone",
         "watch", "speaker", "keyboard", "blender", "kettle"]
CATS = ["electronics", "home", "toys", "sports"]

T_CUST = ["customers", "clients", "buyers", "shoppers"]
T_PROD = ["products", "items", "goods", "catalog"]
T_ORD = ["orders", "purchases", "sales", "transactions"]
C_ID = ["id", "uid", "pk"]
C_NAME = ["name", "full_name", "cname"]
C_CITY = ["city", "town", "region"]
C_AGE = ["age", "years", "yrs"]
C_PNAME = ["pname", "title", "label"]
C_CAT = ["category", "dept", "group_name"]
C_PRICE = ["unit_price", "price", "cost_usd"]
C_QTY = ["qty", "quantity", "units"]
C_CREF = ["customer_id", "client_id", "buyer_ref"]
C_PREF = ["product_id", "item_id", "prod_ref"]


class SqlQueryInstance(TaskInstance):
    task = "sql_query"
    max_turns = 5

    def __init__(self, instance_id: str, seed: int, tier: int = 0):
        super().__init__(instance_id, seed, tier)
        rng = random.Random(seed)
        n = self.n = {
            "t_cust": rng.choice(T_CUST), "t_prod": rng.choice(T_PROD),
            "t_ord": rng.choice(T_ORD),
            "c_id": rng.choice(C_ID), "c_name": rng.choice(C_NAME),
            "c_city": rng.choice(C_CITY), "c_age": rng.choice(C_AGE),
            "c_pname": rng.choice(C_PNAME), "c_cat": rng.choice(C_CAT),
            "c_price": rng.choice(C_PRICE), "c_qty": rng.choice(C_QTY),
            "c_cref": rng.choice(C_CREF), "c_pref": rng.choice(C_PREF),
        }
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        cur = self.db.cursor()
        cur.execute(f"CREATE TABLE {n['t_cust']} ({n['c_id']} INTEGER PRIMARY KEY, "
                    f"{n['c_name']} TEXT, {n['c_city']} TEXT, {n['c_age']} INTEGER)")
        cur.execute(f"CREATE TABLE {n['t_prod']} ({n['c_id']} INTEGER PRIMARY KEY, "
                    f"{n['c_pname']} TEXT, {n['c_cat']} TEXT, {n['c_price']} REAL)")
        cur.execute(f"CREATE TABLE {n['t_ord']} ({n['c_id']} INTEGER PRIMARY KEY, "
                    f"{n['c_cref']} INTEGER, {n['c_pref']} INTEGER, "
                    f"{n['c_qty']} INTEGER)")
        n_cust, n_prod = rng.randint(8, 12), rng.randint(6, 10)
        n_ord = rng.randint(25, 40)
        names = rng.sample(FIRST, n_cust)
        prods = rng.sample(PRODN, n_prod)
        for i in range(n_cust):
            cur.execute(f"INSERT INTO {n['t_cust']} VALUES (?,?,?,?)",
                        (i + 1, names[i], rng.choice(CITY), rng.randint(19, 70)))
        for j in range(n_prod):
            cur.execute(f"INSERT INTO {n['t_prod']} VALUES (?,?,?,?)",
                        (j + 1, prods[j], rng.choice(CATS),
                         round(rng.uniform(4, 400) + rng.random() * 0.9, 2)))
        for k in range(n_ord):
            cur.execute(f"INSERT INTO {n['t_ord']} VALUES (?,?,?,?)",
                        (k + 1, rng.randint(1, n_cust), rng.randint(1, n_prod),
                         rng.randint(1, 6)))
        self.db.commit()
        self.question, self.gold_sql = self._make_question(rng, tier % 3)
        self.gold = self._exec_gold()

    # revenue of an order = qty * unit_price (needs orders x products join)
    def _make_question(self, rng, tier):
        n = self.n
        city = rng.choice(CITY)
        cat = rng.choice(CATS)
        age = rng.randint(25, 55)
        rev = f"o.{n['c_qty']} * p.{n['c_price']}"
        j_op = (f"FROM {n['t_ord']} o JOIN {n['t_prod']} p "
                f"ON o.{n['c_pref']} = p.{n['c_id']}")
        j_all = (j_op + f" JOIN {n['t_cust']} c ON o.{n['c_cref']} = c.{n['c_id']}")
        if tier == 0:
            qs = [
                (f"How many orders are there in total? Answer with a number.",
                 f"SELECT COUNT(*) FROM {n['t_ord']}"),
                (f"How many customers live in {city}? Answer with a number.",
                 f"SELECT COUNT(*) FROM {n['t_cust']} WHERE {n['c_city']}='{city}'"),
                (f"What is the name of the oldest customer? Answer with the name.",
                 f"SELECT {n['c_name']} FROM {n['t_cust']} ORDER BY {n['c_age']} DESC, "
                 f"{n['c_id']} ASC LIMIT 1"),
            ]
        elif tier == 1:
            qs = [
                (f"What is the total revenue (quantity times unit price, summed "
                 f"over all orders) from '{cat}' products? Round to 2 decimals; "
                 f"answer 0 if none.",
                 f"SELECT COALESCE(ROUND(SUM({rev}),2),0) {j_op} "
                 f"WHERE p.{n['c_cat']}='{cat}'"),
                (f"How many distinct customers ever bought a '{cat}' product? "
                 f"Answer with a number.",
                 f"SELECT COUNT(DISTINCT o.{n['c_cref']}) {j_op} "
                 f"WHERE p.{n['c_cat']}='{cat}'"),
                (f"What is the name of the customer whose single order has the "
                 f"highest revenue (quantity times unit price)?",
                 f"SELECT c.{n['c_name']} {j_all} ORDER BY {rev} DESC, "
                 f"o.{n['c_id']} ASC LIMIT 1"),
                (f"How many orders were placed by customers older than {age}? "
                 f"Answer with a number.",
                 f"SELECT COUNT(*) FROM {n['t_ord']} o JOIN {n['t_cust']} c ON "
                 f"o.{n['c_cref']}=c.{n['c_id']} WHERE c.{n['c_age']}>{age}"),
            ]
        else:
            qs = [
                (f"Which product category generates the highest total revenue "
                 f"(quantity times unit price)? Answer with the category name.",
                 f"SELECT p.{n['c_cat']} {j_op} GROUP BY p.{n['c_cat']} "
                 f"ORDER BY SUM({rev}) DESC LIMIT 1"),
                (f"Which city's customers spent the most in total (quantity times "
                 f"unit price)? Answer with the city name.",
                 f"SELECT c.{n['c_city']} {j_all} GROUP BY c.{n['c_city']} "
                 f"ORDER BY SUM({rev}) DESC LIMIT 1"),
                (f"How many customers placed at least 3 orders? Answer with a number.",
                 f"SELECT COUNT(*) FROM (SELECT {n['c_cref']} FROM {n['t_ord']} "
                 f"GROUP BY {n['c_cref']} HAVING COUNT(*) >= 3)"),
                (f"What is the second-highest single-order revenue (quantity times "
                 f"unit price)? Round to 2 decimals.",
                 f"SELECT ROUND({rev},2) {j_op} ORDER BY {rev} DESC "
                 f"LIMIT 1 OFFSET 1"),
                (f"What is the name of the customer with the highest total spend "
                 f"(sum of quantity times unit price over their orders)?",
                 f"SELECT c.{n['c_name']} {j_all} GROUP BY c.{n['c_id']} "
                 f"ORDER BY SUM({rev}) DESC LIMIT 1"),
            ]
        return qs[rng.randrange(len(qs))]

    def _exec_gold(self):
        row = self.db.execute(self.gold_sql).fetchone()
        return row[0] if row else None

    def system_prompt(self):
        return SYSTEM

    def user_prompt(self):
        return f"Question: {self.question}"

    def tools(self):
        def sql(query: str = ""):
            if not query.strip():
                return "SqlError: empty query", True
            try:
                cur = self.db.execute(query)
                rows = cur.fetchmany(20)
                cols = [d[0] for d in cur.description] if cur.description else []
                body = "\n".join(str(tuple(r)) for r in rows) if rows else "(no rows)"
                return f"columns: {cols}\n{body}", False
            except sqlite3.Error as e:
                return f"SqlError: {e}", True
        return {"sql": sql}

    def verify(self, submission):
        gold = self.gold
        cand = submission
        for _ in range(2):
            if isinstance(cand, list) and len(cand) == 1:
                cand = cand[0]
        if isinstance(gold, (int, float)):
            if isinstance(cand, (int, float)):
                return abs(float(cand) - float(gold)) < 0.51
            if isinstance(cand, str):
                import re
                m = re.search(r"-?\d+(?:\.\d+)?", cand.replace(",", ""))
                if m:
                    return abs(float(m.group()) - float(gold)) < 0.51
            return False
        return isinstance(cand, str) and cand.strip().lower() == str(gold).strip().lower()
