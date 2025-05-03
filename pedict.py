import joblib
import numpy as np

# Load the saved model
model = joblib.load('model/hedge_fund_model.pkl')  # Ensure this path is correct

# Get user inputs
stock = int(input("Enter company (encoded) code (e.g., 0 for company 1, 1 for company 2, etc.): "))
accountsPayable = float(input("Enter Accounts Payable: "))
inventory = float(input("Enter Inventory: "))
longTermDebt = float(input("Enter Long Term Debt: "))
netReceivables = float(input("Enter Net Receivables: "))
netTangibleAssets = float(input("Enter Net Tangible Assets: "))
longTermInvestments = float(input("Enter Long Term Investments: "))
totalCurrentAssets = float(input("Enter Total Current Assets: "))
propertyPlantEquipment = float(input("Enter Property, Plant, and Equipment: "))
otherStockholderEquity = float(input("Enter Other Stockholder Equity: "))
deferredLongTermAssetCharges = float(input("Enter Deferred Long-Term Asset Charges: "))
totalCurrentLiabilities = float(input("Enter Total Current Liabilities: "))
cash = float(input("Enter Cash: "))
otherAssets = float(input("Enter Other Assets: "))
treasuryStock = float(input("Enter Treasury Stock: "))
goodWill = float(input("Enter Goodwill: "))
otherLiab = float(input("Enter Other Liabilities: "))
retainedEarnings = float(input("Enter Retained Earnings: "))
otherCurrentAssets = float(input("Enter Other Current Assets: "))
commonStock = float(input("Enter Common Stock: "))
totalAssets = float(input("Enter Total Assets: "))
otherCurrentLiab = float(input("Enter Other Current Liabilities: "))
deferredLongTermLiab = float(input("Enter Deferred Long-Term Liabilities: "))
totalStockholderEquity = float(input("Enter Total Stockholder Equity: "))
totalLiab = float(input("Enter Total Liabilities: "))
capitalSurplus = float(input("Enter Capital Surplus: "))
intangibleAssets = float(input("Enter Intangible Assets: "))
shortTermInvestments = float(input("Enter Short Term Investments: "))
shortLongTermDebt = float(input("Enter Short/Long Term Debt: "))
minorityInterest_x = float(input("Enter Minority Interest (x): "))
capitalExpenditures = float(input("Enter Capital Expenditures: "))
changeToNetincome = float(input("Enter Change to Net Income: "))
otherCashflowsFromFinancingActivities = float(input("Enter Other Cashflows from Financing Activities: "))
changeToAccountReceivables = float(input("Enter Change to Account Receivables: "))
changeToInventory = float(input("Enter Change to Inventory: "))
dividendsPaid = float(input("Enter Dividends Paid: "))
otherCashflowsFromInvestingActivities = float(input("Enter Other Cashflows from Investing Activities: "))
depreciation = float(input("Enter Depreciation: "))
totalCashFromOperatingActivities = float(input("Enter Total Cash from Operating Activities: "))
effectOfExchangeRate = float(input("Enter Effect of Exchange Rate: "))
repurchaseOfStock = float(input("Enter Repurchase of Stock: "))
changeInCash = float(input("Enter Change in Cash: "))
netIncome_x = float(input("Enter Net Income (x): "))
changeToOperatingActivities = float(input("Enter Change to Operating Activities: "))
totalCashFromFinancingActivities = float(input("Enter Total Cash from Financing Activities: "))
netBorrowings = float(input("Enter Net Borrowings: "))
totalCashflowsFromInvestingActivities = float(input("Enter Total Cashflows from Investing Activities: "))
changeToLiabilities = float(input("Enter Change to Liabilities: "))
issuanceOfStock = float(input("Enter Issuance of Stock: "))
investments = float(input("Enter Investments: "))
netIncomeApplicableToCommonShares = float(input("Enter Net Income Applicable to Common Shares: "))
netIncomeFromContinuingOps = float(input("Enter Net Income from Continuing Operations: "))
totalOtherIncomeExpenseNet = float(input("Enter Total Other Income/Expense Net: "))
costOfRevenue = float(input("Enter Cost of Revenue: "))
totalOperatingExpenses = float(input("Enter Total Operating Expenses: "))
totalRevenue = float(input("Enter Total Revenue: "))
incomeTaxExpense = float(input("Enter Income Tax Expense: "))
interestExpense = float(input("Enter Interest Expense: "))
operatingIncome = float(input("Enter Operating Income: "))
ebit = float(input("Enter EBIT: "))
grossProfit = float(input("Enter Gross Profit: "))
sellingGeneralAdministrative = float(input("Enter Selling, General, Administrative: "))
netIncome_y = float(input("Enter Net Income (y): "))
incomeBeforeTax = float(input("Enter Income Before Tax: "))
researchDevelopment = float(input("Enter Research and Development: "))
minorityInterest_y = float(input("Enter Minority Interest (y): "))
discontinuedOperations = float(input("Enter Discontinued Operations: "))

# Create feature array
features = np.array([
    stock, accountsPayable, inventory, longTermDebt, netReceivables, netTangibleAssets,
    longTermInvestments, totalCurrentAssets, propertyPlantEquipment, otherStockholderEquity,
    deferredLongTermAssetCharges, totalCurrentLiabilities, cash, otherAssets, treasuryStock, goodWill,
    otherLiab, retainedEarnings, otherCurrentAssets, commonStock, totalAssets, otherCurrentLiab,
    deferredLongTermLiab, totalStockholderEquity, totalLiab, capitalSurplus, intangibleAssets,
    shortTermInvestments, shortLongTermDebt, minorityInterest_x, capitalExpenditures, changeToNetincome,
    otherCashflowsFromFinancingActivities, changeToAccountReceivables, changeToInventory, dividendsPaid,
    otherCashflowsFromInvestingActivities, depreciation, totalCashFromOperatingActivities, effectOfExchangeRate,
    repurchaseOfStock, changeInCash, netIncome_x, changeToOperatingActivities, totalCashFromFinancingActivities,
    netBorrowings, totalCashflowsFromInvestingActivities, changeToLiabilities, issuanceOfStock, investments,
    netIncomeApplicableToCommonShares, netIncomeFromContinuingOps, totalOtherIncomeExpenseNet, costOfRevenue,
    totalOperatingExpenses, totalRevenue, incomeTaxExpense, interestExpense, operatingIncome, ebit, grossProfit,
    sellingGeneralAdministrative, netIncome_y, incomeBeforeTax, researchDevelopment, minorityInterest_y,
    discontinuedOperations
])

# Reshape features for prediction
features = features.reshape(1, -1)

# Predict
prediction = model.predict(features)

# Output the prediction
if prediction[0] == 1:
    print("Prediction: Profit")
else:
    print("Prediction: Loss")
