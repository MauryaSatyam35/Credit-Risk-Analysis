import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# ── 1. Load the data
df = pd.read_excel("Task 3 and 4_Loan_Data.xlsx")

# ── 2. Define features (X) and target (y)
features = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

X = df[features]
y = df["default"]

# ── 3. Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scale the features (logistic regression works better with scaled data) ─
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 5. Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ── 6. Evaluate the model
y_pred = model.predict(X_test_scaled)
print("=== Model Performance ===")
print(classification_report(y_test, y_pred))


# ── 7. The main function: predict expected loss for a loan
def expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score,
    recovery_rate=0.10,
):
    """
    Given loan details, returns the expected loss.

    Expected Loss = PD  x  Loan Amount  x  (1 - Recovery Rate)

    Parameters:
        credit_lines_outstanding : number of open credit lines
        loan_amt_outstanding     : current loan amount ($)
        total_debt_outstanding   : total debt the borrower has ($)
        income                   : annual income ($)
        years_employed           : years at current job
        fico_score               : credit score
        recovery_rate            : fraction recovered if default (default 10%)

    Returns:
        expected_loss_value (float) : dollar value of expected loss
    """
    loan_features = [[
        credit_lines_outstanding,
        loan_amt_outstanding,
        total_debt_outstanding,
        income,
        years_employed,
        fico_score,
    ]]

    # Scale using the same scaler fitted on training data
    loan_scaled = scaler.transform(loan_features)

    # Probability of default
    pd_prob = model.predict_proba(loan_scaled)[0][1]

    # Expected loss formula
    loss = pd_prob * loan_amt_outstanding * (1 - recovery_rate)

    return pd_prob, loss


# ── 8. Test the function with sample loans
print("\n=== Expected Loss Estimates ===")

# Test 1: Risky borrower (high debt, low income, low FICO)
pd1, loss1 = expected_loss(
    credit_lines_outstanding=5,
    loan_amt_outstanding=5000,
    total_debt_outstanding=25000,
    income=30000,
    years_employed=1,
    fico_score=550,
)
print(f"Risky borrower   -> PD: {pd1:.2%}  |  Expected Loss: ${loss1:,.2f}")

# Test 2: Safe borrower (low debt, high income, high FICO)
pd2, loss2 = expected_loss(
    credit_lines_outstanding=0,
    loan_amt_outstanding=5000,
    total_debt_outstanding=2000,
    income=90000,
    years_employed=6,
    fico_score=750,
)
print(f"Safe borrower    -> PD: {pd2:.2%}  |  Expected Loss: ${loss2:,.2f}")

# Test 3: Average borrower
pd3, loss3 = expected_loss(
    credit_lines_outstanding=2,
    loan_amt_outstanding=4000,
    total_debt_outstanding=9000,
    income=60000,
    years_employed=4,
    fico_score=650,
)
print(f"Average borrower -> PD: {pd3:.2%}  |  Expected Loss: ${loss3:,.2f}")
