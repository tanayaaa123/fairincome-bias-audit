"""
Streamlit dashboard: Bias Audit of an Income Prediction Model
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Fairness Audit", layout="wide")

st.title("Bias & Fairness Audit — Income Prediction Model")
st.caption("Dataset: UCI Adult Income | Protected attributes: sex, race")


# Load precomputed results

with open('results_summary.json') as f:
    results = json.load(f)


# Load & train model (cached)

@st.cache_resource
def load_and_train():
    COLS = [
        'age','workclass','fnlwgt','education','education_num','marital_status',
        'occupation','relationship','race','sex','capital_gain','capital_loss',
        'hours_per_week','native_country','income'
    ]
    df = pd.read_csv('adult.csv', header=None, names=COLS, skipinitialspace=True)
    df['income'] = df['income'].str.strip().str.rstrip('.')
    df['income'] = (df['income'] == '>50K').astype(int)
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    df = df.replace('?', np.nan).dropna()

    df_model = df.copy()
    cat_cols = ['workclass','education','marital_status','occupation',
                'relationship','race','sex','native_country']
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_model[c] = le.fit_transform(df_model[c])
        encoders[c] = le

    feature_cols = [c for c in df_model.columns if c not in ['income']]
    X = df_model[feature_cols]
    y = df_model['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    return df, clf, scaler, encoders, feature_cols

df, clf, scaler, encoders, feature_cols = load_and_train()


# TABS

tab1, tab2, tab3 = st.tabs(["Audit Summary", "Try a Prediction", "Proxy Variable Problem"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy (before)", f"{results['overall_acc_before']*100:.1f}%")
    col2.metric("Demographic Parity Gap (before)", f"{results['demographic_parity_gap_before']*100:.1f}%")
    col3.metric("Equal Opportunity Gap (before)", f"{results['equal_opportunity_gap_before']*100:.1f}%")

    st.subheader("Selection rate by sex — before vs after mitigation")
    sex_before = results['sex_audit_before']
    sex_after = results['sex_audit_after']

    fig, ax = plt.subplots(figsize=(6,3.5))
    groups = list(sex_before.keys())
    before_rates = [sex_before[g]['selection_rate'] for g in groups]
    after_rates = [sex_after[g]['selection_rate'] for g in groups]
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width/2, before_rates, width, label='Before mitigation', color='#d62728')
    ax.bar(x + width/2, after_rates, width, label='After mitigation', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('Selection rate (% predicted >50K)')
    ax.legend()
    ax.set_title('Model approves men at a much higher rate than women — before fixing')
    st.pyplot(fig)

    st.subheader("The tradeoff")
    st.write(f"""
    - Accuracy before mitigation: **{results['overall_acc_before']*100:.1f}%**
    - Accuracy after mitigation: **{results['overall_acc_after']*100:.1f}%**
    - Cost of fixing the bias: **{(results['overall_acc_before']-results['overall_acc_after'])*100:.1f} percentage points**
    - Demographic parity gap dropped from **{results['demographic_parity_gap_before']*100:.1f}%** to **{results['demographic_parity_gap_after']*100:.1f}%**
    """)
    st.info("This is the honest finding: reducing bias here has a real, measurable accuracy cost. "
            "Whether that tradeoff is acceptable is a business/ethics decision, not a technical one.")

    st.subheader("Accuracy & selection rate by race (unmitigated)")
    race_df = pd.DataFrame(results['race_audit_before']).T
    st.dataframe(race_df)

with tab2:
    st.write("Enter a profile and see the model's prediction, using the ORIGINAL (unmitigated) model.")
    c1, c2, c3 = st.columns(3)
    age = c1.slider("Age", 17, 90, 35)
    hours = c2.slider("Hours per week", 1, 99, 40)
    edu_num = c3.slider("Education (years, numeric)", 1, 16, 10)

    sex = c1.selectbox("Sex", df['sex'].unique())
    race = c2.selectbox("Race", df['race'].unique())
    relationship = c3.selectbox("Relationship", df['relationship'].unique())

    workclass = c1.selectbox("Workclass", df['workclass'].unique())
    marital = c2.selectbox("Marital status", df['marital_status'].unique())
    occupation = c3.selectbox("Occupation", df['occupation'].unique())

    if st.button("Predict"):
        row = df.iloc[0].copy()
        row['age'] = age
        row['hours_per_week'] = hours
        row['education_num'] = edu_num
        row['sex'] = sex
        row['race'] = race
        row['relationship'] = relationship
        row['workclass'] = workclass
        row['marital_status'] = marital
        row['occupation'] = occupation

        enc_row = row.copy()
        for c, le in encoders.items():
            enc_row[c] = le.transform([row[c]])[0]

        X_input = enc_row[feature_cols].values.reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        prob = clf.predict_proba(X_scaled)[0][1]

        st.metric("Predicted probability of income >50K", f"{prob*100:.1f}%")
        st.write(f"Prediction: **{'>50K' if prob >= 0.5 else '<=50K'}**")

        # quick fairness flag
        if sex == 'Female' and prob < 0.5:
            st.warning("Note: the audit shows this model systematically under-predicts high income "
                       "for female profiles even with similar age/education/hours. Interpret this "
                       "prediction with that context.")

with tab3:
    st.subheader("How 'relationship' leaks sex, even without using the sex column directly")
    crosstab = pd.crosstab(df['relationship'], df['sex'], normalize='index')
    st.dataframe(crosstab.style.format("{:.1%}"))
    st.write("""
    Categories like **Husband** and **Wife** almost perfectly separate male and female records.
    This means: even if you remove the `sex` column entirely from training, the model can still
    reconstruct gender through `relationship` and discriminate indirectly. This is called a
    **proxy variable** — it's one of the most common ways real-world ML systems end up biased
    even after "fixing" the obvious protected attribute.
    """)
