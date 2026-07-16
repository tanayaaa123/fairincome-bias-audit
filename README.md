# Bias & Fairness Audit of an Income Prediction Model


I built a model that predicts if someone earns more than $50K/year, then checked
if the model treats men and women (and different races) unfairly — and found that
it does. Then I fixed it, and measured what that fix cost in accuracy.




## The dataset
UCI Adult Income dataset — real US Census data from 1994. About 45,000 people.
For each person we know: age, education, job type, hours worked, marital status,
sex, race, etc. The task: predict if they earn more than $50K/year.







### Step 1 — Cleaned the data
Removed rows with missing values, cleaned up text formatting, converted the target
(income) into a simple 0/1 (0 = earns ≤$50K, 1 = earns >$50K).

### Step 2 — Trained a normal model, no fairness considerations
Used Logistic Regression (a standard, simple classification model) to predict
income. This is the "baseline" — a model built the normal way, the way most
people build models without thinking about fairness at all.

**Result: 81.8% accuracy.** Sounds good on its own.

### Step 3 — Checked if the model treats groups differently
This is the actual "audit" part. I split the test data by sex and by race, and
for each group I checked two things:

- **Selection rate** — what % of this group did the model predict as high income?
  If the model is fair, this should be roughly similar across groups.
- **True positive rate** — among people in this group who ACTUALLY earn >$50K,
  what % did the model correctly identify?

**What I found:**
- Model predicted 21.3% of men as high earners, but only 3.3% of women.
- That's an 18-percentage-point gap. This is a big, real bias.
- Among people who genuinely earn >$50K, the model correctly caught 49% of men
  but only 19.5% of women. So the model isn't just reflecting reality — it's
  actively worse at recognizing successful women.

### Step 4 — Found WHY the bias exists — a hidden leak
Here's the part most beginners miss. I checked: even if I never told the model
someone's sex, could it still figure it out from other columns?

Answer: yes. There's a column called "relationship" (values like Husband, Wife,
Unmarried, Own-child). It turns out "Husband" is 99.99% male and "Wife" is 99.95%
female. So even without a "sex" column, the model can basically reconstruct
someone's sex through "relationship" and discriminate anyway.

This is called a **proxy variable** — a column that indirectly reveals a protected
attribute like sex or race. It's the reason "just delete the sensitive column" does
NOT actually fix bias in real systems. This is the single most important insight
in the whole project — mention it in interviews.

### Step 5 — Fixed the bias (a bit) and measured the cost
I used a technique called **threshold adjustment**. Here's what that means simply:

Normally, a model predicts "high income" if its confidence score is above 50%.
That single 50% cutoff is the same for everyone. Instead, I set a DIFFERENT cutoff
for men and women — a lower cutoff for women, a higher one for men — so that both
groups end up getting approved at roughly the same rate.

**Result after the fix:**
- The 18% gap between men and women dropped to basically 0.1% — nearly fixed.
- But overall accuracy dropped from 81.8% to 79.4% — a real cost of 2.4 points.

This is the most important finding of the whole project: **making a model fairer
is not free. It costs accuracy.** Whether that tradeoff is worth it is a business
decision, not something I can answer with code — but showing this tradeoff clearly
is exactly what a company would want to see.

---

## Files in this project
- `fairness_audit.py` — the full pipeline. Run this first: `python3 fairness_audit.py`
  It cleans data, trains the model, runs the audit, applies the fix, and saves
  the results to `results_summary.json`.
- `app.py` — a simple visual dashboard (Streamlit) showing all of the above, plus
  a page where you can enter a fake profile and see what the model predicts.
  Run with: `streamlit run app.py`
- `adult.csv` — the dataset
- `results_summary.json` — saved numbers from the audit, used by the dashboard

## How to run it
pip install pandas numpy scikit-learn shap streamlit --break-system-packages
python3 fairness_audit.py
streamlit run app.py
---




