import pandas as pd
import matplotlib.pyplot as plt
import pickle

#reads file, sheet at index = 0
df = pd.read_excel("ensembl-export-serpina1.xlsx", sheet_name=0)

#replace all Clin. Sig. with uncertain significance with VUS
df["Clin. Sig."] = df["Clin. Sig."].mask(
    df["Clin. Sig."].str.contains("uncertain significance", case=False, na=False),
    "VUS"
)

#replace all Clin. Sig. with benign as benign
df["Clin. Sig."] = df["Clin. Sig."].mask(
    df["Clin. Sig."].str.contains("benign", case=False, na=False),
    "benign"
)

#replace all Clin. Sig. with pathogenic as pathogenic
df["Clin. Sig."] = df["Clin. Sig."].mask(
    df["Clin. Sig."].str.contains("pathogenic", case=False, na=False),
    "pathogenic"
)

#remove all observations labeled as "not provided" or "other"
df = df[~df["Clin. Sig."].isin(["not provided", "other"])]

#fill all na with conditional average, normalize SIFT
df['SIFT'] = df['SIFT'].fillna(df.groupby("Clin. Sig.")['SIFT'].transform('mean'))
df["SIFT"] = 1 - df["SIFT"]
df['PolyPhen'] = df['PolyPhen'].fillna(df.groupby("Clin. Sig.")['PolyPhen'].transform('mean'))
df['REVEL'] = df['REVEL'].fillna(df.groupby("Clin. Sig.")['REVEL'].transform('mean'))
df['MetaLR'] = df['MetaLR'].fillna(df.groupby("Clin. Sig.")['MetaLR'].transform('mean'))
df['Mutation Assessor'] = df['Mutation Assessor'].fillna(df.groupby("Clin. Sig.")['Mutation Assessor'].transform('mean'))

#drop all na
df = df.dropna(subset=["Clin. Sig.", "SIFT", "PolyPhen", "REVEL", "MetaLR", "Mutation Assessor"])

#get index of 196 random benign to get even sample size for both categories
rows = df[df["Clin. Sig."] == "benign"].sample(n=196, random_state=42).index

#drop them
df2 = df.drop(rows)

#save data frame of full sample and dropped sample using pickle
file = open('DATA_CLEANED.pkl', 'wb')
pickle.dump(df2, file)
file.close()

file = open('DATA_CLEANED_FULL.pkl', 'wb')
pickle.dump(df, file)
file.close()

print("Done")
